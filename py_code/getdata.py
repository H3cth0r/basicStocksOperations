import yfinance as yf
import pandas as pd
import numpy as np

# Getting the data
data_nvidia = yf.download("NVDA", start='2021-06-19', end='2022-06-19')
data_amd = yf.download("AMD", start='2021-06-19', end='2022-06-19')

# Saving data to csv
data_nvidia.to_csv("./csvs/nvidia_data.csv")
data_amd.to_csv("./csvs/amd_data.csv")