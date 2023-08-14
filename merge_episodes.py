import argparse
import gzip
import json
from pathlib import Path

from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument("--episodes-dir", type=Path)
parser.add_argument("--out-path", type=Path)

def main(episodes_dir: Path, out_path: Path):
    episodes_files = episodes_dir.glob("*.gz")
    all_episodes = {}
    cat_fields = [
        "obj_category_to_obj_category_id",
        "recep_category_to_recep_category_id",
    ]
    for episodes_file in tqdm(episodes_files):
        with gzip.open(episodes_file) as f:
            episodes = json.load(f)
        if not all_episodes:
            all_episodes = episodes
        else:
            all_episodes["episodes"].extend(episodes["episodes"])
        if cat_fields[0] not in all_episodes:
            for field in cat_fields:
                all_episodes[field] = episodes[field]

    for i, episode in enumerate(all_episodes["episodes"]):
        episode["episode_id"] = str(i)

    for field in cat_fields:
        print(all_episodes[field])

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with gzip.open(out_path, "wt") as f:
        f.write(json.dumps(all_episodes))

if __name__ == "__main__":
    args = parser.parse_args()
    main(**vars(args))
