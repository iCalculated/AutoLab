import seaborn as sns 
from matplotlib import pyplot as plt
import pandas as pd 

df = pd.read_csv('data.csv', index_col=0)
print(df.head(5))