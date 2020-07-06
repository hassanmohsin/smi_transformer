import pandas as pd
import numpy as np
df = pd.read_csv("data/250k_rndm_zinc_drugs_clean_3.csv")
props = ['logP', 'qed', 'sas']
norms = pd.DataFrame({'mean': df[props].mean(axis=0), 'std': df[props].std(axis=0)})
df[props] = (df[props] - norms['mean']) / norms['std']

df.to_csv("data/250k_rndm_zinc_drugs_clean_3_normalized.csv", index=None)
norms.to_csv("data/norms.csv")
