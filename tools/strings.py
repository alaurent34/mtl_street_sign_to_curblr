# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 09:02:40 2021

@author: lgauthier
"""

def is_numeric_transformable(val):
    try:
        float(val)
        return True
    except:
        return False