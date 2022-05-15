import numpy as np
import pandas as pd
import sys



def find_min():
    df = pd.read_csv(sys.argv[1], skipfooter=1, engine='python')
    minMSE_pos = df['MSE_pos'].min()
    print(f"MIN MSE_pos: {minMSE_pos}")
    parameters = df.loc[df['MSE_pos'] == minMSE_pos]
    hs = parameters['hidden_size'].values[0]
    all_hs = df.loc[df['hidden_size'] == hs].sort_values(by=['MSE_pos']).head(10)
    print(all_hs)


if __name__ == "__main__":
    find_min()
