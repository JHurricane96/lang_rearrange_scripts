import argparse
import json
import gzip
from pathlib import Path

parser = argparse.ArgumentParser()

parser.add_argument("--episodes-file")
parser.add_argument("--out-dir", type=Path)

def main(args):
    with gzip.open(args.episodes_file) as f:
        dataset_str = f.read()
        dataset = json.loads(dataset_str)

    episodes = dataset["episodes"]
    split_episodes = {}
    for episode in episodes:
        scene_id = Path(episode["scene_id"]).name.split(".", 1)[0]
        if scene_id not in split_episodes:
            split_episodes[scene_id] = {"episodes": []}
        split_episodes[scene_id]["episodes"].append(episode)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    for k, v in dataset.items():
        if k == "episodes":
            continue
        for scene, split in split_episodes.items():
            split[k] = v

            with gzip.open(args.out_dir/f"{scene}.json.gz", "wt") as f:
                json.dump(split, f)

if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
