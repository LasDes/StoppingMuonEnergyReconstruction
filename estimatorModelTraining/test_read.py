import pandas as pd

FILE = "../ftable_11499.hdf5"
df_att = pd.read_hdf(FILE, "HoinkaAttributes")
df_lbl = pd.concat([pd.read_hdf(FILE, "HoinkaLabels")],
                   axis=1)
print(df_lbl)