'''
created: August 15, 2014
last update: June 6 2015
version 0.2.5
James Houghton <james.p.houghton@gmail.com>
'''

# pysd specific imports
import translators as _translators

# third party imports
import pandas as _pd
import numpy as np
import imp
from numba import jit


# Todo: Check that having the init be a function attribute is ok, because sometimes it will be a function,
#       and we may not be ready to run the function at import time.
#
# Todo: add a logical way to run two or more models together, using the same integrator.
# Todo: import translators within read_XMILE and read_Vensim, so we don't load both if we dont need them

######################################################


######################################################
# Issues:
#
# If two model components (A, and B) depend on the same third model component (C)
# then C will get computed twice. If C itself is dependant on many upstream components (D, E, F...)
# then these will also be computed multiple times.
#
# As the model class depends on its internal state for calculating
# the values of each element, instead of passing that state
# through the function execution network, we can't use caching
# to prevent multiple execution, as we don't know when to update the cache
# (maybe in the calling function?)
#
# Also, this multi-calculation bears the risk that if the state is
# changed during an execution step, the resulting calculation will be
# corrupted, and we won't have any indication of this corruption.
######################################################

def read_xmile(xmile_file):
    """ Construct a model object from `.xmile` file. """
    py_model_file = _translators.translate_xmile(xmile_file)
    model = load(py_model_file)
    model.__str__ = 'Import of ' + xmile_file
    return model


read_xmile.__doc__ += _translators.translate_xmile.__doc__


def read_vensim(mdl_file):
    """ Construct a model from Vensim `.mdl` file. """
    py_model_file = _translators.translate_vensim(mdl_file)
    model = load(py_model_file)
    model.__str__ = 'Import of ' + mdl_file
    return model





def load(py_model_file):
    """ Load a python-converted model file. """
    components = imp.load_source('modulename', py_model_file)
    components._stocknames = [name for name in dir(components) if hasattr(getattr(components, name), 'init')]
    components._dfuncs = {name: getattr(components, 'd%s_dt' % name) for name in components._stocknames}

    isfunc = lambda x: not x.startswith('_') and hasattr(getattr(components, x), '__call__')
    funcnames = filter(isfunc, dir(components))

    # apply the @JIT to all functions
    [setattr(components, f, jit(getattr(components, f))) for f in funcnames]
    components._funcs = {name: getattr(components, name) for name in funcnames}

    @jit
    def _step(dt):
        newstate = {}
        for key in components._stocknames:
            newstate[key] = components._dfuncs[key]() * dt + components.state[key]
        components.state = newstate
        components.t = components.t+dt

    @jit()
    def _integrate(timesteps, return_elements):
        outputs = range(len(timesteps))
        for i, t2 in enumerate(timesteps):
            _step(t2 - components.t)
            outdict = {}
            for key in return_elements:
                outdict[key] = components._funcs[key]()
            outputs[i] = outdict
        return outputs

    setattr(components, '_step', _step)
    setattr(components, '_integrate', _integrate)

    model = PySD(components)
    model.__str__ = 'Import of ' + py_model_file
    return model


class PySD(object):
    """
        PySD is the default class charged with running a model.

        It can be initialized by passing an existing component class.

        The import functions pull models and create this class.
    """

    def __init__(self, components):
        """ Construct a PySD object built around the component class """
        self.components = components
        self.record = []

    def __str__(self):
        """ Return model source file """
        return self.components.__str__

    def run(self, params={}, return_columns=[], return_timestamps=[],
            initial_condition='original', collect=False, **intg_kwargs):
        """ Simulate the model's behavior over time.
        Return a pandas dataframe with timestamps as rows,
        model elements as columns.

        Parameters
        ----------

        params : dictionary
            Keys are strings of model component names.
            Values are numeric or pandas Series.
            Numeric values represent constants over the model integration.
            Timeseries will be interpolated to give time-varying input.

        return_timestamps : list, numeric, numpy array(1-D)
            Timestamps in model execution at which to return state information.
            Defaults to model-file specified timesteps.

        return_columns : list of string model component names
            Returned dataframe will have corresponding columns.
            Defaults to model stock values.

        initial_condition : 'original'/'o', 'current'/'c', (t, {state})
            The starting time, and the state of the system (the values of all the stocks)
            at that starting time.

            * 'original' (default) uses model-file specified initial condition
            * 'current' uses the state of the model after the previous execution
            * (t, {state}) lets the user specify a starting time and (possibly partial)
              list of stock values.

        collect: binary (T/F)
            When running multiple simulations, collect the results in a way
            that we can access down the road.

        intg_kwargs: keyword arguments for odeint
            Provides precice control over the integrator by passing through
            keyword arguments to scipy's odeint function. The most interesting
            of these will be `tcrit`, `hmax`, `mxstep`.


        Examples
        --------

        >>> model.run(params={'exogenous_constant':42})
        >>> model.run(params={'exogenous_variable':timeseries_input})
        >>> model.run(return_timestamps=[1,2,3.1415,4,10])
        >>> model.run(return_timestamps=10)
        >>> model.run(return_timestamps=np.linspace(1,10,20))

        See Also
        --------
        pysd.set_components : handles setting model parameters
        pysd.set_initial_condition : handles setting initial conditions

        """

        if params:
            self.set_components(params)

        if initial_condition != 'current':
            self.set_initial_condition(initial_condition)

        tseries = self._build_timeseries(return_timestamps)

        # the odeint expects the first timestamp in the tseries to be the initial condition,
        # so we may need to add the t0 if it is not present in the tseries array
        addtflag = tseries[0] != self.components.t
        if addtflag:
            tseries = np.insert(tseries, 0, self.components.t)

        if self.components._stocknames:
            # res = _odeint(func=self.components.d_dt,
            #              y0=self.components.get_state(),
            #              t=tseries,
            #              **intg_kwargs)
            #              #hmax=self.components.time_step())

            if not return_columns:
                return_columns = self.components._stocknames

            res = self.components._integrate(tseries, return_columns)

            return_df = _pd.DataFrame(data=res,
                                      index=tseries,
                                      columns=return_columns)  # ,

        else:
            outdict = {}
            for key in return_columns:
                outdict[key] = self.components._funcs[key]()
            return_df = _pd.DataFrame(index=tseries, data=outdict)

        if addtflag:
            return_df.drop(return_df.index[0], inplace=True)

        if collect:
            self.record.append(return_df)  # we could just record the state, and expand it later...

        return return_df

    def reset_state(self):
        """Sets the model state to the state described in the model file. """
        self.components.t = self.components.initial_time()  # set the initial time
        self.components.state = {}
        retry_flag = False
        for key in self.components._stocknames:
            try:
                # We have to do a loop here because there are cases where the initialization will
                # call a function, and that function may not have its own initial conditions defined
                # just yet.
                self.components.state[key] = getattr(self.components, key).init  # set the initial state
            except TypeError:
                retry_flag = True
        if retry_flag:
            self.reset_state()  # potential for infinite loop!

    def get_record(self):
        """ Return the recorded model information.
        Returns everything as a big long dataframe.

        >>> model.get_record()
        """
        return _pd.concat(self.record)

    def clear_record(self):
        """ Reset the recorder.

        >>> model.clear_record()
        """
        self.record = []

    def set_components(self, params):
        """ Set the value of exogenous model elements.
        Element values can be passed as keyword=value pairs in the function call.
        Values can be numeric type or pandas Series.
        Series will be interpolated by integrator.

        Examples
        --------
        >>> br = pandas.Series(index=range(30), values=np.sin(range(30))
        >>> set_components(birth_rate=br)
        >>> set_components(birth_rate=10)

        """
        updates_dict = {}
        for key, value in params.iteritems():
            if isinstance(value, _pd.Series):
                updates_dict[key] = self._timeseries_component(value)
            else:  # could check here for valid value...
                updates_dict[key] = self._constant_component(value)

        self.components.__dict__.update(updates_dict)

    def set_state(self, t, state):
        """ Set the system state.

        t : numeric
            The system time

        state: dict
            Idelly a complete dictionary of system state, but a partial
            state dictionary will work if you're confident that the remaining
            state elements are correct.
        """
        self.components.t = t
        self.components.state.update(state)

    def set_initial_condition(self, initial_condition):
        """ Set the initial conditions of the integration.
        There are several ways to do this:

        * 'original'/'o' : Reset to the model-file specified initial condition.
        * 'current'/'c' : Use the current state of the system to start
          the next simulation. This includes the simulation time, so this
          initial condition must be paired with new return timestamps
        * (t, {state}) : Lets the user specify a starting time and list of stock values.

        >>> model.set_initial_condition('original')
        >>> model.set_initial_condition('current')
        >>> model.set_initial_condition( (10,{teacup_temperature:50}) )

        See also:
        pysd.set_state()
        """

        if isinstance(initial_condition, tuple):
            # we should probably check the values more than just seeing if they are a tuple.
            self.set_state(*initial_condition)
        elif isinstance(initial_condition, str):
            if initial_condition.lower() in ['original', 'o']:
                self.reset_state()
            elif initial_condition.lower() in ['current', 'c']:
                pass
            else:
                raise ValueError('Valid initial condition strings include:  \n' +
                                 '    "original"/"o",                       \n' +
                                 '    "current"/"c"')
        else:
            raise TypeError('Check documentation for valid entries')

    def _build_timeseries(self, return_timestamps):
        """ Build up array of timestamps """
        if return_timestamps == []:
            tseries = np.arange(self.components.initial_time(),
                                self.components.final_time(),
                                self.components.time_step())
        elif isinstance(return_timestamps, (list, int, float, long, np.ndarray)):
            tseries = np.array(return_timestamps, ndmin=1)
        else:
            raise TypeError('`return_timestamps` expects a list, array, or numeric value')
        return tseries

    # these could be better off in a model creation class
    def _timeseries_component(self, series):
        """ Internal function for creating a timeseries model element """
        return lambda: np.interp(self.components.t, series.index, series.values)

    def _constant_component(self, value):
        """ Internal function for creating a constant model element """
        return lambda: value

#
# @jit
# def _step(ddt, state, dt):
#     outdict = {}
#     for key in ddt:
#         outdict[key] = ddt[key]() * dt + state[key]
#     return outdict
#
#
# @jit(nopython=True)
# def _integrate(components, ddt, timesteps, return_elements):
#     outputs = range(len(timesteps))
#     for i, t2 in enumerate(timesteps):
#         components.state = components._step(ddt, components.state, t2 - components.t)
#         components.t = t2
#         outdict = {}
#         for key in return_elements:
#             outdict[key] = components._funcs[key]()
#         outputs[i] = outdict
#
#     return outputs
