import argparse
import json
import os

import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_path", type=str)
args = parser.parse_args()

with open(os.path.join(args.dataset_path, "transforms_train.json"), "r") as f:
    data_train = json.load(f)
with open(os.path.join(args.dataset_path, "transforms_test.json"), "r") as f:
    data_test = json.load(f)
with open(os.path.join(args.dataset_path, "transforms_val.json"), "r") as f:
    data_val = json.load(f)

assert data_train["fl_x"] == data_val["fl_x"]
assert data_train["fl_y"] == data_val["fl_y"]
assert data_train["h"] == data_val["h"]
assert data_train["w"] == data_val["w"]
assert data_train["cx"] == data_val["cx"]
assert data_train["cy"] == data_val["cy"]
assert data_train["camera_angle_x"] == data_val["camera_angle_x"]
assert data_train["camera_angle_y"] == data_val["camera_angle_y"]

data_splits = {'train': data_train, 'val': data_val, 'test': data_test}

for k, data_split in data_splits.items():
    frames = data_split['frames']
    for i, f in enumerate(tqdm(frames)):
        flame_param_fname = os.path.join(args.dataset_path, f["flame_param_path"])
        fp = np.load(flame_param_fname)
        data_split["frames"][i]["exp_ori"] = fp["expr"].tolist()[0]

    with open(os.path.join(args.dataset_path, f'transforms_nb_{k}.json'), "w") as f:
        json.dump(data_split, f)
    print(f"Saved {k} split to {os.path.join(args.dataset_path, f'transforms_nb_{k}.json')}")


len_val = "-" + str(len(data_val["frames"]))

# Save len_val to a tempfile to late read in with bash script
if os.path.exists("/tmp/len_val"):
    os.remove("/tmp/len_val")

with open("/tmp/len_val", "w") as f:
    f.write(len_val)
