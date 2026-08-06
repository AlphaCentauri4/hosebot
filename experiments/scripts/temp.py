import pandas as pd 
import matplotlib.pyplot as plt


# df = pd.read_csv("data\\experiment_data\\20260730_155525.csv")
df = pd.read_csv("data\\experiment_data\\ygdfdgg.csv")
# df = df[:int(len(df)/3)]
plt.plot(df["flow_left%"], df["flow_right%"])
plt.title("Flow left v. flow_right")
plt.xlabel("flow_left")
plt.ylabel("flow_right")
plt.show()