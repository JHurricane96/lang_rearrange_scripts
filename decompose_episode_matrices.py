import argparse
import gzip
import json
from collections import Counter, defaultdict
from pathlib import Path
import time

import numpy as np
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("--out-dir", type=Path)
parser.add_argument("--episodes-path")

args = parser.parse_args()

out_dir = args.out_dir
episodes_path = args.episodes_path

# with gzip.open("cat_npz-exp.json.gz") as f:
#     dataset = json.load(f)
# vp_matrix = np.load("cat_npz-exp-vps.npy")
# trans_matrix = np.load("cat_npz-exp-trans.npy")
# import pdb; pdb.set_trace()

# with gzip.open("cat_fp_10-ep.json.gz") as f:
start_t = time.time()
with gzip.open(episodes_path) as f:
    f_str = f.read()
    load_t = time.time()
    print(f"Loaded file in {load_t - start_t}s.")
    dataset = json.loads(f_str)
parse_t = time.time()
print(f"Parsed JSON in {parse_t - load_t}s.")

vp_keys = ["candidate_objects", "candidate_objects_hard", "candidate_goal_receps"]

def target_check(episode):
    for target in episode["targets"]:
        rigid_objs_count = Counter(r[0].split(".", 1)[0] for r in episode["rigid_objs"])
        target_name, target_num = target.split(":")
        target_name = target_name[:-1]
        target_num = int(target_num) + 1
        if target_num != 1:
            return False
        if rigid_objs_count[target_name] < target_num:
            return False
    return True

seen_rec_vps = defaultdict(dict)
vp_matrix = []
trans_matrix = []
eps_to_remove = []
for i, ep in tqdm(enumerate((dataset["episodes"]))):
    if not target_check(ep):
        eps_to_remove.append(i)
        print("Removing target check failed episode", i)
        continue
    scene_id = ep["scene_id"]
    for obj in ep["rigid_objs"]:
        trans_matrix.append(obj[1][:3])
        obj[1] = len(trans_matrix) - 1
    for vp_key in vp_keys:
        for obj in ep[vp_key]:
            object_name = obj["object_name"]
            if vp_key == "candidate_goal_receps" and object_name in seen_rec_vps[scene_id]:
                cached_vps = seen_rec_vps[scene_id][object_name]
                assert len(obj["view_points"]) == len(cached_vps)
                assert obj["view_points"][0]["iou"] == vp_matrix[cached_vps[0]][-1]
                obj["view_points"] = seen_rec_vps[scene_id][object_name]
                continue
            vp_idxs = []
            for vp in obj["view_points"]:
                vp_matrix.append(vp["agent_state"]["position"] + vp["agent_state"]["rotation"] + [vp["iou"]])
                vp_idxs.append(len(vp_matrix) - 1)
            obj["view_points"] = vp_idxs
            if vp_key == "candidate_goal_receps" and object_name not in seen_rec_vps[scene_id]:
                seen_rec_vps[scene_id][object_name] = vp_idxs

for i in reversed(eps_to_remove):
    del dataset["episodes"][i]

print("New dataset length:", len(dataset["episodes"]))

out_dir.mkdir(parents=True, exist_ok=True)
vp_matrix = np.array(vp_matrix, dtype=np.float32)
trans_matrix = np.array(trans_matrix, dtype=np.float32)
with gzip.open(out_dir/"cat_npz-exp.json.gz", "wt") as f:
    f.write(json.dumps(dataset))
np.save(out_dir/"cat_npz-exp-vps.npy", vp_matrix)
np.save(out_dir/"cat_npz-exp-trans.npy", trans_matrix)
