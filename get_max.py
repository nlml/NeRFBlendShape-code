import json
import numpy as np
import os
import ipdb

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--path", type=str)
parser.add_argument("--num", type=int)

opt = parser.parse_args()
path = opt.path


with open(path, "r") as f:
    meta = json.load(f)

exps = []
for frame in meta["frames"]:
    exps.append(frame["exp_ori"])


num = opt.num
exps = np.array(exps, dtype=np.float32)

max_per = np.max(exps, axis=0)
max_per_savepath = os.path.join(os.path.dirname(path), "max_" + str(num) + ".txt")
np.savetxt(max_per_savepath, max_per[:num])
print("saved to " + max_per_savepath)

min_per = np.min(exps, axis=0)
min_per_savepath = os.path.join(os.path.dirname(path), "min_" + str(num) + ".txt")
np.savetxt(min_per_savepath, min_per[:num])
print("saved to " + min_per_savepath)
