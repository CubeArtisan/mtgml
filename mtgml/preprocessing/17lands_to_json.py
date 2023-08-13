import collections
import collections.abc
import itertools
import json
import sys

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


def row_to_tuple(row: pd.Series) -> tuple[int, ...]:
    return tuple(itertools.chain.from_iterable(itertools.repeat(int(col), int(val)) for col, val in row.items()))


def get_pick(row, name_to_int, num_picks):
    cards_in_pack = row_to_tuple(row.iloc[4:])
    return {
        "cardsInPack": cards_in_pack,
        "pickedIdx": cards_in_pack.index(name_to_int[row.iloc[3].lower()]),
        "packNum": int(row.iloc[1]),
        "pickNum": int(row.iloc[2]),
        "numPicks": num_picks,
        "numPacks": 3,
    }


def get_draft(rows, basics, name_to_int):
    num_picks = len(rows) // 3
    try:
        return {
            "basics": basics,
            "picks": [
                get_pick(row, name_to_int, num_picks)
                for _, row in rows.sort_values(["pack_number", "pick_number"]).iterrows()
            ],
        }
    except ValueError:
        print("Draft contained invalid pick.")
        return {}


def get_deck(
    row: pd.Series, main_cols: list[tuple[str, int]], side_cols: list[tuple[str, int]], basics: tuple[int, ...]
):
    main = row[[col for col, _ in main_cols]]
    main.index = [idx for _, idx in main_cols]
    side = row[[col for col, _ in side_cols]]
    side.index = [idx for _, idx in side_cols]
    return {
        "main": row_to_tuple(main),
        "side": row_to_tuple(side),
        "basics": basics,
    }


def read_draft_file(name_to_int, basics, filename):
    prefix = "pack_card_"
    data = pd.read_csv(filename, engine="pyarrow")
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
        get_draft(data.iloc[idxs], basics, name_to_int)
        for _, idxs in tqdm(
            data.groupby("draft_id", sort=True).groups.items(),
            dynamic_ncols=True,
            unit="draft",
            unit_scale=1,
            smoothing=0.01,
        )
        if len(idxs) == 42 or len(idxs) == 45 or len(idxs) == 39
    )


def read_game_file(name_to_int, basics, filename):
    data = pd.read_csv(filename, engine="pyarrow")
    main_prefix = "deck_"
    main_cols = [
        (col, name_to_int[col.removeprefix(main_prefix).lower()]) for col in data.columns if col.startswith(main_prefix)
    ]
    side_prefix = "sideboard_"
    side_cols = [
        (col, name_to_int[col.removeprefix(side_prefix).lower()]) for col in data.columns if col.startswith(side_prefix)
    ]
    index_cols = ["draft_id", "build_index"]
    decks_df = data.set_index(index_cols)[[col for col, _ in main_cols + side_cols]].groupby(index_cols).first()
    return (
        get_deck(row, main_cols, side_cols, basics)
        for _, row in tqdm(
            decks_df.iterrows(), total=decks_df.shape[0], dynamic_ncols=True, unit="deck", unit_scale=1, smoothing=0.01
        )
    )


if __name__ == "__main__":
    with open("data/maps/int_to_card.json") as fp:
        int_to_card = json.load(fp)
    name_to_int = {c["name"].lower(): i for i, c in enumerate(int_to_card)}
    basics = tuple(name_to_int[c] for c in ["plains", "island", "swamp", "mountain", "forest"])
    fname = sys.argv[1].split("/")[-2]
    if fname == "draft":
        results = read_draft_file(name_to_int, basics, sys.argv[1])
    elif fname == "game":
        results = read_game_file(name_to_int, basics, sys.argv[1])
    else:
        raise ValueError(f"Unknown file type for {sys.argv[1]}")
    with open(f"{sys.argv[1]}.json", "w") as fp:
        json.dump(results, fp=fp, cls=IterEncoder)
