import pandas as pd 
import matplotlib.pyplot as plt


df = pd.read_csv("experiment_data\\20260730_155525.csv")
plt.plot(df["A1%"], 1-df["A2%"]+df["A3%"])
plt.show()