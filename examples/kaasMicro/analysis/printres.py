import pandas as pd
import sys

dat = pd.read_csv(sys.argv[1], header=0, comment="#")
dat.set_index('name', inplace=True)

for idx, row in dat.iterrows():
    print("=== " + idx + " ===")
    print(row.to_string())
    print('')
