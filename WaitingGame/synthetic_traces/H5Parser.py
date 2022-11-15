#!/usr/bin/env python3

import pandas as pd

df = pd.read_hdf("synthetic-trace-1.h5")
df = df.head(10000)

df.to_hdf("sampled.h5", "df")
df.to_csv("sampled.csv", index=False)