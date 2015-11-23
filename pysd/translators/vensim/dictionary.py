"""
dictionary.py

This is the vensim to python translation dictionary. Many expressions (+-/*, etc)
translate directly, but some need to have their names changed, etc. This is
how we decide that. If it isn't in the dictionary, it probably won't work

"""

dictionary = {"ABS":"abs",
              "INTEGER":"int",
              "EXP":"np.exp",
              "PI":"np.pi",
              "SIN":"np.sin",
              "COS":"np.cos",
              "SQRT":"np.sqrt",
              "TAN":"np.tan",
              "LOGNORMAL":"np.random.lognormal",
              "RANDOM NORMAL":"self.functions.bounded_normal",
              "POISSON":"np.random.poisson",
              "LN":"np.log",
              "EXPRND":"np.random.exponential",
              "RANDOM UNIFORM":"np.random.rand",
              "MIN":"min",
              "MAX":"max",
              "ARCCOS":"np.arccos",
              "ARCSIN":"np.arcsin",
              "ARCTAN":"np.arctan",
              "IF THEN ELSE":"self.functions.if_then_else",
              "STEP":"self.functions.step",
              "MODULO":"np.mod",
              "PULSE":"self.functions.pulse",
              "PULSE TRAIN":"self.functions.pulse_train",
              "RAMP":"self.functions.ramp",
              "=":"==",
              "<=":"<=",
              "<":"<",
              ">=":">=",
              ">":">",
              "^":"**",
              ":AND:": "and",
              ":OR:":"or",
              ":NOT:":"not"}