import pandas as pd


data = pd.read_csv("./data/bruh.csv", index_col="Gmt time")
data['Volume'] = data["Volume"] * 1_000_000
data.to_csv("./data/temp.csv")
