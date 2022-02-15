# -*- coding: utf-8 -*-
"""
Created on Sat Jun 19 00:50:04 2021

@author: lgauthier
"""
import warnings
import functools

import numpy as np

def xform_list_entre_to_dict(listValues, keyList, default=None, valueVarName='valueVarName'):
    if not listValues:
        listValues = {key:0 for key in keyList}
    elif isinstance(listValues, (tuple, list, np.ndarray)):
        listValues = {keyList[k]:listValues[k] for k in range(len(keyList))}
    else:
        raise TypeError("`valueVarName` must be None or one of {{tuple, list, np.ndarray}}, received {listValues.__class__}")
    return listValues


def checkraise(asked, valids, argname):
    if isinstance(asked, (list, tuple, np.ndarray)):
        check = np.asarray([elem in valids for elem in asked]).all()
        if not check:
            raise ValueError(f"The argument '{argname}' must only contain "+
                              "elements whose values are in "+
                             f"{{{', '.join([str(v) for v in valids])}}}"+
                             f", received: {asked}")

    if not asked in valids:
        raise ValueError(f"The argument '{argname}' must be one of "+
                         f"{{{', '.join([str(v) for v in valids])}}}"+
                         f", received: {asked}")

##########################
###### EXCEPTIONS ########
##########################

class IncompleteDataError(Exception):
    """An exception signaling that an expected column in a dataframe is missing.
    """
    def __init__(self, df, *column, context=None):
        if len(column) == 1:
            message = f"Missing column `{column[0]}` for dataframe `{df}`."
        else:
            column = [f"`{c}`" for c in column]
            message = f"Missing columns {', '.join(column)} for dataframe `{df}`."

        if context is not None:
            message += f' {context}'
        self.message = message

        super().__init__(message)


##########################
###### WARNINGS ##########
##########################

class MissingDataWarning(UserWarning):
    """A warning signaling that some data is missing to fully complete the
    required analysis.
    """
    def __init__(self, message):
        super().__init__(message)
        self.message = message

def deprecated(func):
    """This is a decorator which can be used to mark functions
    as deprecated. It will result in a warning being emitted
    when the function is used."""
    @functools.wraps(func)
    def new_func(*args, **kwargs):
        warnings.simplefilter('always', DeprecationWarning)  # turn off filter
        warnings.warn("Call to deprecated function {}.".format(func.__name__),
                      category=DeprecationWarning,
                      stacklevel=2)
        warnings.simplefilter('default', DeprecationWarning)  # reset filter
        return func(*args, **kwargs)
    return new_func
