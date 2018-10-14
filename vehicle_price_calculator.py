# -*- coding: utf-8 -*-
"""
Created on Sun Oct 14 17:01:28 2018

@author: Notorious-V
"""

import tensorflow as tf
import csv
import pandas as pd

def main():
    
    #set display options
    pd.set_option('display.max_colwidth', 100)
    
    #read in data file
    data = pd.read_csv('CARS.csv')
    print(data)
main()