import tensorflow as tf
import pandas as pd
from numpy import array
from numpy import argmax
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from keras.utils import to_categorical



def main():
    dataset = pd.read_csv('drive/My Drive/app/CARS.csv')
    df = pd.DataFrame(dataset)

    print(df)
main()