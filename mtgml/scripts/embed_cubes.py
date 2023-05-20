import glob
import json
import locale
import logging

import numpy as np
from jsonslicer import JsonSlicer
from tqdm.auto import tqdm

from mtgml.constants import MAX_CUBE_SIZE
from mtgml.server import get_model

locale.setlocale(locale.LC_ALL, "en_US.UTF-8")

with open("data/maps/int_to_card.json") as fp:
    int_to_card = json.load(fp)
name_to_int = {card["name_lower"]: i for i, card in enumerate(int_to_card)}


def pad(arr, desired_length, value=0):
    if isinstance(arr, tuple):
        if len(arr) < desired_length:
            arr = list(arr)
        else:
            return arr[:desired_length]
    return arr + [value for _ in range(desired_length - len(arr))]


def load_all_cubes(cube_dirs):
    num_cubes = 0
    for cube_dir in cube_dirs.split(";"):
        for cubes_file in tqdm(
            glob.glob(f"{cube_dir}/*.json"), leave=False, dynamic_ncols=True, unit="file", unit_scale=1
        ):
            try:
                with open(cubes_file, "rb") as fp:
                    cubes = JsonSlicer(fp, (None,))
                    for cube in tqdm(
                        cubes,
                        leave=False,
                        dynamic_ncols=True,
                        unit="cube",
                        unit_scale=1,
                        smoothing=0.001,
                        initial=num_cubes,
                    ):
                        if MAX_CUBE_SIZE >= len(cube["cards"]) >= 120 and all(
                            isinstance(x, int) for x in cube["cards"]
                        ):
                            num_cubes += 1
                            yield (pad(tuple(x + 1 for x in cube["cards"]), MAX_CUBE_SIZE), cube["name"], cube["id"])
            except:
                logging.exception(f"Error in file {cubes_file}")
    print(f"Total cubes {num_cubes:n}")


def load_all_cubes2(cube_dirs):
    num_cubes = 0
    for cube_dir in cube_dirs.split(";"):
        for cubes_file in tqdm(
            glob.glob(f"{cube_dir}/*.json"), leave=False, dynamic_ncols=True, unit="file", unit_scale=1
        ):
            with open(cubes_file, "rb") as fp:
                cubes = JsonSlicer(fp, (None,))
                for cube in tqdm(
                    cubes,
                    leave=False,
                    dynamic_ncols=True,
                    unit="cube",
                    unit_scale=1,
                    smoothing=0.001,
                    initial=num_cubes,
                ):
                    try:
                        if MAX_CUBE_SIZE >= len(cube["cards"]) >= 120 and all(x in name_to_int for x in cube["cards"]):
                            num_cubes += 1
                            yield (
                                pad(tuple(name_to_int[x] + 1 for x in cube["cards"]), MAX_CUBE_SIZE),
                                cube["name"],
                                cube["id"],
                            )
                    except:
                        continue


if __name__ == "__main__":
    model = get_model()
    with open("data/cube_embeddings.csv", "w") as embed_handle, open("data/cube_data.csv", "w") as data_handle:
        data_handle.write("name,id\n")
        cubes = load_all_cubes2("data/CubeCobra/cubes")
        for cards, name, id in cubes:
            embedding = model.cube_recommender((np.array([cards]), model.embed_cards.embeddings), training=False)[
                1
            ].numpy()[0]
            embed_handle.write(",".join(str(x) for x in embedding) + "\n")
            name = name.replace('"', '""')
            data_handle.write(f'"{name}",https://cubecobra.com/cube/list/{id}\n')
        cubes = load_all_cubes("data/20220829/cubes")
        for cards, name, id in cubes:
            embedding = model.cube_recommender((np.array([cards]), model.embed_cards.embeddings), training=False)[
                1
            ].numpy()[0]
            embed_handle.write(",".join(str(x) for x in embedding) + "\n")
            name = name.replace('"', '""')
            data_handle.write(f'"{name}",https://cubeartisan.net/cube/{id}/list\n')
