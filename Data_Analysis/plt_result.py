# -*- coding: utf-8 -*-
"""
Created on Sun Feb 18 22:54:07 2018

@author: smili
"""

from DataProcessing import ToPandasData
from DataVisulization import VisMeasurements
test_data = ToPandasData('../Data_Collection/' + 'aaa' + '.txt')
VisMeasurements(test_data)
