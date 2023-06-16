import argparse
import json
from pathlib import Path
import shutil
import yaml

from tqdm import tqdm

parser = argparse.ArgumentParser()

parser.add_argument("--split-config", default="habitat-lab/habitat/datasets/rearrange/configs/rearrange_floorplanner.yaml")
parser.add_argument("--objects-config", default="data/objects/google_scanned/google_scanned.scene_dataset_config.json", type=Path)
parser.add_argument("--new-objects-dir", default="data/objects_new", type=Path)


def main(args):
    with open(args.split_config) as f:
        split_config = yaml.full_load(f)
    with open(args.objects_config) as f:
        dataset_config = json.load(f)

    objs_dir: Path = args.objects_config.parent
    configs_dir = Path(dataset_config["objects"]["paths"][".json"][0])
    dataset = objs_dir.stem

    with tqdm() as pbar:
        for obj_set in split_config["object_sets"]:
            set_name = obj_set["name"]
            set_dest_dir = args.new_objects_dir
            if set_name.startswith("test_"):
                set_dest_dir /= "test"
            else:
                set_dest_dir /= "train_val"
            set_dest_dir /= dataset

            for obj in obj_set["included_substrings"]:
                obj_config_file = f"{obj}object_config.json"
                obj_config_path = objs_dir/configs_dir/obj_config_file

                if not obj_config_path.exists():
                    continue

                with open(obj_config_path) as f:
                    obj_config = json.load(f)

                render_asset_file = obj_config_path.parent / obj_config["render_asset"]
                coll_asset_file = obj_config_path.parent / obj_config["collision_asset"]

                config_dest = set_dest_dir/configs_dir/obj_config_file
                coll_dest = config_dest.parent/obj_config["collision_asset"]
                if render_asset_file.suffix == ".obj":
                    render_dest = (config_dest.parent/obj_config["render_asset"]).parent.parent
                else:
                    render_dest = config_dest.parent/obj_config["render_asset"]

                config_dest.parent.mkdir(exist_ok=True, parents=True)
                coll_dest.parent.mkdir(exist_ok=True, parents=True)
                render_dest.parent.mkdir(exist_ok=True, parents=True)

                shutil.copy(obj_config_path, config_dest)
                shutil.copy(render_asset_file, render_dest)
                if render_asset_file.suffix == ".obj":
                    shutil.copytree(render_asset_file.parent.parent, render_dest, dirs_exist_ok=True)
                else:
                    shutil.copy(coll_asset_file, coll_dest)

                pbar.update()
    
    objects_config_file = args.objects_config.name
    shutil.copy(args.objects_config, args.new_objects_dir/"train_val"/dataset/objects_config_file)
    shutil.copy(args.objects_config, args.new_objects_dir/"test"/dataset/objects_config_file)


if __name__ == "__main__":
    args = parser.parse_args()
    main(args)
