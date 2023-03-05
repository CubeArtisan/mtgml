import collections
import collections.abc
import csv
import itertools
import json
import sys
from pathlib import Path

import pandas as pd
from tqdm.auto import tqdm


class IterEncoder(json.JSONEncoder):
    """
    JSON Encoder that encodes iterators as well.
    Write directly to file to use minimal memory
    """

    class FakeListIterator(list):
        def __init__(self, iterable):
            self.iterable = iter(iterable)
            try:
                self.firstitem = next(self.iterable)
                self.truthy = True
            except StopIteration:
                self.truthy = False

        def __iter__(self):
            if not self.truthy:
                return iter([])
            return itertools.chain([self.firstitem], self.iterable)

        def __len__(self):
            raise NotImplementedError("Fakelist has no length")

        def __getitem__(self, i):
            raise NotImplementedError("Fakelist has no getitem")

        def __setitem__(self, i):
            raise NotImplementedError("Fakelist has no setitem")

        def __bool__(self):
            return self.truthy

    def default(self, o):
        if isinstance(o, collections.abc.Iterable):
            return type(self).FakeListIterator(o)
        return super().default(o)


def get_pick(row, name_to_int):
    cards_in_pack = tuple(
        itertools.chain.from_iterable(itertools.repeat(col, val) for col, val in row.iloc[4:].items())
    )
    return {
        "cardsInPack": cards_in_pack,
        "pickedIdx": cards_in_pack.index(name_to_int[row.iloc[3].lower()]),
        "packNum": int(row.iloc[1]),
        "pickNum": int(row.iloc[2]),
        "numPicks": 14,
        "numPacks": 3,
    }


def get_draft(rows, name_to_int):
    try:
        return {
            "picks": [
                get_pick(row, name_to_int) for _, row in rows.sort_values(["pack_number", "pick_number"]).iterrows()
            ]
        }
    except ValueError:
        return {}


def read_file(name_to_int, filename):
    prefix = "pack_card_"
    data = pd.read_csv(filename)
    pack_cols = [col for col in data.columns if col.startswith(prefix)]
    old_cols = ["draft_id", "pack_number", "pick_number", "pick", *pack_cols]
    new_cols = [
        "draft_id",
        "pack_number",
        "pick_number",
        "pick",
        *[name_to_int[col.removeprefix(prefix).lower()] for col in pack_cols],
    ]
    data = data[old_cols]
    data.columns = new_cols
    return (
        get_draft(data.iloc[idxs], name_to_int)
        for _, idxs in tqdm(
            data.groupby("draft_id", sort=True).groups.items(),
            dynamic_ncols=True,
            unit="draft",
            unit_scale=1,
            smoothing=0.01,
        )
        if len(idxs) == 42
    )


if __name__ == "__main__":
    with open("data/maps/int_to_card.json") as fp:
        int_to_card = json.load(fp)
    name_to_int = {c["name"].lower(): i for i, c in enumerate(int_to_card)}
    drafts = read_file(name_to_int, sys.argv[1])
    with open(f"{sys.argv[1]}.json", "w") as fp:
        json.dump(drafts, fp=fp, cls=IterEncoder)
