import os 
import sys
import pandas as pd
print(os.system('pwd'))
df = pd.read_csv('GBP_EUR Historische Data.csv')
print(df[0])
