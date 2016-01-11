########## Templates #############

from string import Template

templates = {}

templates['new file'] = Template(
"""
from __future__ import division
import numpy as np
from pysd import functions
""")

templates['stock'] = Template(
"""
def ${identifier}():
    return state['${identifier}']
${identifier}.init = ${initial_condition}

def d${identifier}_dt():
    return ${expression}
""")

templates['flaux'] = Template(
"""
def ${identifier}():
    \"""
    ${docstring}
    \"""
    return ${expression}
""")
