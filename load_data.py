import numpy as np
import pandas as pd
import os

data_path = './outlier/train102.csv'

def read_data(data_path):
    df = pd.read_csv(data_path)
    value = np.array(df['Value'])
    label = np.array(df['Label'])
    return value, label

def normalize_data(value):
    mean = np.mean(value)
    std = np.std(value)
    # print(mean)
    # print(std)
    value = [(x - mean) / std for x in value]
    return value

def create_window(value, label, WINDOW_SIZE):
    windows = np.array([value[:WINDOW_SIZE]])
    for i in range(len(value) - WINDOW_SIZE):
        windows = np.concatenate((windows, np.array([value[i + 1:i + 1 + WINDOW_SIZE]])))
    label = label[WINDOW_SIZE - 1:]
    label = label[:,np.newaxis]
    return windows, label

def save_np(filename, windows, label):
    np.savez(filename, x=windows, y=label)

def load_np(filename):
    f = np.load(filename)
    x = f['x']
    y = f['y']
    return x, y

if __name__ == '__main__':
    value, label = read_data(data_path)
    value = normalize_data(value)
    windows, label = create_window(value, label, 128)
    save_np('train102', windows, label)