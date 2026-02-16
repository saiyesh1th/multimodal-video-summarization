import pandas as pd
import glob
import numpy as np
import matplotlib.pyplot as plt

files = glob.glob("mfcc_sample/part-*.json")
df = pd.concat([pd.read_json(f, lines=True) for f in files])

mfcc = np.vstack(df["mfcc"].values)

plt.imshow(mfcc.T, aspect="auto", origin="lower", cmap="magma")
plt.colorbar(label="MFCC value")
plt.xlabel("Time (5s chunks)")
plt.ylabel("MFCC coefficient")
plt.title("MFCC Heatmap")
plt.show()
