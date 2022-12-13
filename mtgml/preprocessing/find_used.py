import glob
import json
import locale
import logging
import multiprocessing.pool
import random
import struct
import sys
from collections import Counter
from pathlib import Path
from typing import Generator, Iterable

import numpy as np
import zstandard as zstd
from jsonslicer import JsonSlicer
from tqdm.auto import tqdm

from mtgml.constants import MAX_BASICS, MAX_CARDS_IN_PACK, MAX_PICKED, MAX_SEEN_PACKS, MAX_CUBE_SIZE
from mtgml.utils.grid import interpolate

locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

with open('data/maps/card_to_int.json') as fp:
    card_to_int = json.load(fp)
with open('data/maps/int_to_card.json') as fp:
    int_to_card = json.load(fp)
with open('data/maps/old_int_to_card.json') as fp:
    old_int_to_card = json.load(fp)
old_int_to_new_int = [card_to_int[c["oracle_id"]] if c['oracle_id'] in card_to_int else -1 for c in old_int_to_card]
name_to_int = {card['name_lower']: i for i, card in enumerate(int_to_card)}
default_basic_ids = [
    "56719f6a-1a6c-4c0a-8d21-18f7d7350b68",
    "b2c6aa39-2d2a-459c-a555-fb48ba993373",
    "bc71ebf6-2056-41f7-be35-b2e5c34afa99",
    "b34bb2dc-c1af-4d77-b0b3-a0fb342a5fc6",
    "a3fb7228-e76b-4e96-a40e-20b5fed75685",
]
default_basics = tuple(card_to_int[c] + 1 for c in default_basic_ids)

BUFFER_SIZE = 1024 // 64


ALL_PICKS = tuple(0 for _ in range(MAX_SEEN_PACKS))


def picks_from_draft2(draft) -> Generator[Counter[int], None, None]:
    if 'picks' in draft:
        seen = []
        seen_indices = set()
        for pick in draft['picks']:
            if not (all(isinstance(x, int) for x in pick['cardsInPack'])
                    and all(isinstance(x, int) for x in pick['picked'])
                    and len(seen) < MAX_SEEN_PACKS
                    and len(pick['picked']) <= MAX_PICKED):
                break
            cards_in_pack = [old_int_to_new_int[x] + 1 for x in pick['cardsInPack']]
            # coords, coord_weights = interpolate(pick['pick'], pick['packSize'], pick['pack'],
            #                                     pick['packs'])
            chosen_card = old_int_to_new_int[pick['chosenCard']] + 1
            picked_idx = cards_in_pack.index(chosen_card)
            if picked_idx < 0:
                break
            cards_in_pack[0], cards_in_pack[picked_idx] = cards_in_pack[picked_idx], cards_in_pack[0]
            cards_in_pack = cards_in_pack[:MAX_CARDS_IN_PACK]
            seen.append(())
            # seen.append(cards_in_pack[:MAX_CARDS_IN_PACK])
            seen_indices |= set(cards_in_pack[:MAX_CARDS_IN_PACK])
            # seen_coords.append(coords)
            # seen_coord_weights.append(coord_weights)
            # value = tuple(old_int_to_new_int[x] + 1 for x in pick['picked'])
        yield Counter(seen_indices)


def picks_from_draft(draft) -> Generator[Counter[int], None, None]:
    if isinstance(draft, dict):
        basics = [x + 1 for x in draft.get('basics', default_basics)][:MAX_BASICS]
        if 'picks' in draft:
            seen = []
            seen_indices = set(basics)
            for pick in draft['picks']:
                picked_idx = pick.get('pickedIdx', pick.get('trashedIdx', None))
                if not (all(isinstance(x, int) for x in pick['cardsInPack'])
                        and all(isinstance(x, int) for x in pick['picked'])
                        and len(seen) < MAX_SEEN_PACKS
                        and picked_idx is not None and 0 <= picked_idx < len(pick['cardsInPack'])
                        and len(pick['picked']) <= MAX_PICKED):
                    break
                cards_in_pack = [x + 1 for x in pick['cardsInPack']]
                cards_in_pack[0], cards_in_pack[picked_idx] = cards_in_pack[picked_idx], cards_in_pack[0]
                cards_in_pack = cards_in_pack[:MAX_CARDS_IN_PACK]
                seen_indices |= set(cards_in_pack)
            yield Counter(seen_indices)


def load_all_drafts(pool, *args) -> Generator[Counter[int], None, None]:
    for drafts_dir, picks_gen in zip(args, (picks_from_draft, picks_from_draft2)):
        def load_drafts_file(drafts_file) -> Generator[Counter[int], None, None]:
            try:
                with open(drafts_file, 'rb') as fp:
                    drafts = JsonSlicer(fp, (None,))

                    def gen() -> Generator[Counter[int], None, None]:
                        for draft in tqdm(drafts, leave=False, dynamic_ncols=True, unit='draft', unit_scale=1,
                                          smoothing=0.001):
                            yield from picks_gen(draft)
                    yield from gen()
            except:
                logging.exception(f'Error in file {drafts_file}')
        for draft_dir in drafts_dir.split(';'):
            files = glob.glob(f'{draft_dir}/*.json')
            for sets in tqdm((load_drafts_file(file) for file in files),
                             leave=False, dynamic_ncols=True, unit='file', unit_scale=1, total=len(files)):
                yield from sets

def load_all_cubes(cube_dirs) -> Generator[Counter[int], None, None]:
    num_cubes = 0
    for cube_dir in cube_dirs.split(';'):
        for cubes_file in tqdm(glob.glob(f'{cube_dir}/*.json'), leave=False, dynamic_ncols=True,
                                unit='file', unit_scale=1):
            try:
                with open(cubes_file, 'rb') as fp:
                    cubes = JsonSlicer(fp, (None,))
                    for cube in tqdm(cubes, leave=False, dynamic_ncols=True, unit='cube', unit_scale=1,
                                      smoothing=0.001, initial=num_cubes):
                        if MAX_CUBE_SIZE >= len(cube['cards']) >= 120 and all(isinstance(x, int) for x in cube['cards']):
                            num_cubes += 1
                            yield Counter(set(x + 1 for x in cube['cards']))
            except:
                logging.exception(f'Error in file {cubes_file}')
    print(f'Total cubes {num_cubes:n}')


def load_all_cubes2(cube_dirs) -> Generator[Counter[int], None, None]:
    num_cubes = 0
    for cube_dir in cube_dirs.split(';'):
        for cubes_file in tqdm(glob.glob(f'{cube_dir}/*.json'), leave=False, dynamic_ncols=True,
                                unit='file', unit_scale=1):
            try:
                with open(cubes_file, 'rb') as fp:
                    cubes = JsonSlicer(fp, (None, 'cards'))
                    for cube in tqdm(cubes, leave=False, dynamic_ncols=True, unit='cube', unit_scale=1,
                                      smoothing=0.001, initial=num_cubes):
                        if MAX_CUBE_SIZE >= len(cube) >= 120 and all(x in name_to_int for x in cube):
                            num_cubes += 1
                            yield Counter(set(name_to_int[x] + 1 for x in cube))
            except:
                logging.exception(f'Error in file {cubes_file}')
    print(f'Total cubes {num_cubes:n}')


if __name__ == '__main__':
    with multiprocessing.pool.ThreadPool(processes=1) as read_pool:
        seen_indices = Counter()
        for seen_set in load_all_drafts(read_pool, sys.argv[1], sys.argv[2]):
            seen_indices += seen_set
        seen_indices_1 = seen_indices
        seen_indices = Counter()
        for seen_set in load_all_cubes(sys.argv[3]):
            seen_indices += seen_set
        for seen_set in load_all_cubes2(sys.argv[4]):
            seen_indices += seen_set
        seen_indices = seen_indices + seen_indices_1
        for i in range(1, 1001):
            clipped_seen_indices = [k for k, v in seen_indices.items() if v >= i]
            print(i, len(clipped_seen_indices))
        seen_indices = [k for k, v in seen_indices.items() if v >= 100]
        new_map = [0 for _ in int_to_card] + [0]
        for new_index, original_index in enumerate(sorted(seen_indices)):
            new_map[original_index] = new_index
        with open('data/maps/original_to_new_index.json', 'w') as fp:
            json.dump(new_map, fp)
        print('Reduced from', len(int_to_card) + 1, 'cards to', max(new_map) + 1)
