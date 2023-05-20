import json
from pathlib import Path

import numpy as np
from mtgml_native.generators.draftbot_generator import DraftbotGenerator
from mtgml_native.generators.recommender_generator import RecommenderGenerator

DATA_DIR = Path("data/")
MODEL_DIR = Path("ml_files/latest/")

with open(MODEL_DIR / "int_to_card.json") as fp:
    INT_TO_CARD = json.load(fp)
NAME_TO_INT = {c["name_lower"]: i + 1 for i, c in enumerate(INT_TO_CARD)}
INT_TO_NAME = [""] + [c["name_lower"] for c in INT_TO_CARD]

ADJ_MTX = np.load(MODEL_DIR / "adj_mtx.npy")


def get_draftbot_gen():
    gen = DraftbotGenerator(str(DATA_DIR / "train_picks.bin"), len(INT_TO_CARD) + 1, 1, 37)
    gen.__enter__()
    return gen


def get_adj_sums(choices, pool):
    return ADJ_MTX[choices][:, pool].sum(axis=-1)


def get_sample(gen):
    return [x[0] for x in gen[0]]
