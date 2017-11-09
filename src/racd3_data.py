#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov  9 11:25:20 2017

@author: Axa
"""

import pandas as pd
import numpy as np


def load_racd3_data():

    """Charge the radcd3 dataset and outputs a matrix X and the countdata Y"""
    
    variables = ['SEX',
                 'AGE',
                 'AGESQ',
                 'INCOME',
                 'LEVYPLUS',
                 'FREEPOOR',
                 'FREEREPA',
                 'ILLNESS'	,
                 'ACTDAYS'	,
                 'HSCORE',
                 'CHCOND1'	,
                 'CHCOND2']

    output = ['DVISITS']

    X = pd.read_excel('/Users/Axa/Documents/Adel/Stochastic Gradient descent/BDD/racd3.xlsx',
                      convert_float=False,
                      usecols=variables,
                      header=0)
    X = np.array(X)
    indices = np.random.randint(len(X), size=len(X))
    X_2 = np.take(X, indices=indices, axis=0)

    Y = pd.read_excel('/Users/Axa/Documents/Adel/Stochastic Gradient descent/BDD/racd3.xlsx',
                      convert_float=False,
                      usecols=output,
                      header=0)

    Y = np.array(Y)
    Y_2 = np.take(Y, indices=indices, axis=0)

    return(X_2, Y_2)
