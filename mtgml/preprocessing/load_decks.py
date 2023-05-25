import glob
import json
import locale
import logging
import random
import struct
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from jsonslicer import JsonSlicer
from tqdm.auto import tqdm

from mtgml.constants import MAX_COPIES, MAX_DECK_SIZE

locale.setlocale(locale.LC_ALL, "en_US.UTF-8")

with open("data/maps/int_to_card.json") as fp:
    int_to_card = json.load(fp)
with open("data/maps/card_to_int.json") as fp:
    card_to_int = json.load(fp)
original_to_new_path = Path("data/maps/original_to_new_index.json")
if original_to_new_path.exists():
    with original_to_new_path.open("r") as fp:
        original_to_new_index = json.load(fp)
else:
    original_to_new_index = tuple(range(len(card_to_int) + 1))
new_to_original = {v: i for i, v in enumerate(original_to_new_index)}
new_to_original[0] = 0
cobra_path = Path("data/CubeCobra/indexToOracleMap.json")
if cobra_path.exists():
    with cobra_path.open() as fp:
        cobra_int_to_oracle = json.load(fp)
    cobra_to_new_index = [
        original_to_new_index[card_to_int.get(cobra_int_to_oracle[str(i)], -1) + 1]
        for i in range(max([int(x) for x in cobra_int_to_oracle.keys()]) + 1)
    ]
else:
    cobra_to_new_index = defaultdict(lambda: -1)

is_land_old = [False] + ["Land" in card["type"] for card in int_to_card]
is_land = np.array(
    [1 if is_land_old[new_to_original[i]] else 0 for i in range(max(original_to_new_index) + 1)], dtype=np.int32
)


def pad(arr, desired_length, value=0):
    if isinstance(arr, tuple):
        if len(arr) < desired_length:
            arr = list(arr)
        else:
            return arr[:desired_length]
    return arr + [value for _ in range(desired_length - len(arr))]


DESTS = [0, 0, 0, 0, 0, 0, 0, 0, 1, 2]


def load_all_decks(deck_dirs):
    num_decks = 0
    for deck_dir in deck_dirs.split(";"):
        for decks_file in tqdm(
            glob.glob(f"{deck_dir}/*.json"), leave=False, dynamic_ncols=True, unit="file", unit_scale=1
        ):
            try:
                with open(decks_file, "rb") as fp:
                    decks = JsonSlicer(fp, (None,))
                    for deck in tqdm(
                        decks,
                        leave=False,
                        dynamic_ncols=True,
                        unit="deck",
                        unit_scale=1,
                        smoothing=0.001,
                        initial=num_decks,
                    ):
                        if (
                            len(deck["main"]) == 40
                            and all(isinstance(x, int) for x in deck["main"])
                            and len(deck["side"]) > 0
                            and len(deck["main"]) + len(deck["side"]) <= MAX_DECK_SIZE
                            and all(isinstance(x, int) for x in deck["side"])
                        ):
                            rand_val = random.randint(0, 9)
                            dest = DESTS[rand_val]
                            main = [original_to_new_index[x + 1] for x in deck["main"]]
                            side = [original_to_new_index[x + 1] for x in deck["side"]]
                            pool = main + side
                            if all(x > 0 for x in pool) and 15 <= is_land[main].sum() <= 18:
                                pool_counter = Counter(pool)
                                remaining_space = MAX_DECK_SIZE - len(pool)
                                num_basics = len(deck["basics"])
                                max_to_add = 0
                                for i, basic in enumerate([original_to_new_index[x + 1] for x in deck["basics"]]):
                                    max_to_add = MAX_COPIES - pool_counter.get(basic, 0)
                                    if max_to_add < 0:
                                        break
                                    pool += [
                                        basic
                                        for _ in range(
                                            min(
                                                max_to_add,
                                                (i + 1) * remaining_space // num_basics,
                                            )
                                        )
                                    ]
                                if max_to_add >= 0:
                                    pool = sorted(pool)
                                    pool_counter = defaultdict(int)
                                    copy_counts = []
                                    for card in pool:
                                        copy_counts.append(pool_counter[card])
                                        pool_counter[card] += 1
                                    if max(pool_counter.values()) >= MAX_COPIES:
                                        continue
                                    indices = {card: i for i, card in list(enumerate(pool))[::-1]}
                                    target = [0 for _ in pool]
                                    for card in main:
                                        target[indices[card]] = 1
                                        indices[card] += 1
                                    num_decks += 1
                                    yield (dest, (pool, copy_counts, target))
            except:
                logging.exception(f"Error in file {decks_file}")
    print(f"Total decks {num_decks:n}")


def load_all_old_decks(deck_dirs):
    num_decks = 0
    for deck_dir in deck_dirs.split(";"):
        for decks_file in tqdm(
            glob.glob(f"{deck_dir}/*.json"), leave=False, dynamic_ncols=True, unit="file", unit_scale=1
        ):
            try:
                with open(decks_file, "rb") as fp:
                    decks = JsonSlicer(fp, (None,))
                    for deck in tqdm(
                        decks,
                        leave=False,
                        dynamic_ncols=True,
                        unit="deck",
                        unit_scale=1,
                        smoothing=0.001,
                        initial=num_decks,
                    ):
                        if (
                            len(deck["mainboard"]) == 40
                            and all(isinstance(x, int) for x in deck["mainboard"])
                            and len(deck["sideboard"]) > 0
                            and len(deck["mainboard"]) + len(deck["sideboard"]) <= MAX_DECK_SIZE
                            and all(isinstance(x, int) for x in deck["sideboard"])
                        ):
                            rand_val = random.randint(0, 9)
                            dest = DESTS[rand_val]
                            main = [cobra_to_new_index[x] for x in deck["mainboard"]]
                            side = [cobra_to_new_index[x] for x in deck["sideboard"]]
                            pool = main + side
                            if all(x > 0 for x in pool) and 15 <= is_land[main].sum() <= 18:
                                pool_counter = Counter(pool)
                                remaining_space = MAX_DECK_SIZE - len(pool)
                                num_basics = len(deck["basics"])
                                max_to_add = 0
                                for i, basic in enumerate([cobra_to_new_index[x] for x in deck["basics"]]):
                                    max_to_add = MAX_COPIES - pool_counter.get(basic, 0)
                                    if max_to_add < 0:
                                        break
                                    to_add = min(
                                        max_to_add,
                                        (i + 1) * remaining_space // num_basics - i * remaining_space // num_basics,
                                    )
                                    pool += [basic for _ in range(to_add)]
                                    pool_counter[basic] += to_add
                                if max_to_add >= 0:
                                    pool = sorted(pool)
                                    pool_counter = defaultdict(int)
                                    copy_counts = []
                                    for card in pool:
                                        copy_counts.append(pool_counter[card])
                                        pool_counter[card] += 1
                                    if max(pool_counter.values()) > MAX_COPIES:
                                        continue
                                    indices = {card: i for i, card in list(enumerate(pool))[::-1]}
                                    target = [0 for _ in pool]
                                    for card in main:
                                        target[indices[card]] = 1
                                        indices[card] += 1
                                    num_decks += 1
                                    yield (dest, (pool, copy_counts, target))
            except:
                logging.exception(f"Error in file {decks_file}")
    print(f"Total decks {num_decks:n}")


PREFIX = struct.Struct(f"{MAX_DECK_SIZE}H{MAX_DECK_SIZE}H{MAX_DECK_SIZE}H")


def write_deck(deck, output_file):
    prefix = PREFIX.pack(*pad(deck[0], MAX_DECK_SIZE), *pad(deck[1], MAX_DECK_SIZE), *pad(deck[2], MAX_DECK_SIZE))
    output_file.write(prefix)


if __name__ == "__main__":
    train_filename = Path("data/train_decks.bin")
    validation_filename = Path("data/validation_decks.bin")
    evaluation_filename = Path("data/evaluation_decks.bin")
    with open(train_filename, "wb") as train_file, open(validation_filename, "wb") as validation_file, open(
        evaluation_filename, "wb"
    ) as evaluation_file:
        output_files = [train_file, validation_file, evaluation_file]
        for dest, deck in load_all_decks(sys.argv[1]):
            write_deck(deck, output_files[dest])
        for dest, deck in load_all_old_decks(sys.argv[2]):
            write_deck(deck, output_files[dest])
