import glob
import json
import locale
import logging
import multiprocessing.pool
import random
import struct
import sys
from pathlib import Path

import numpy as np
import zstandard as zstd
from jsonslicer import JsonSlicer
from tqdm.auto import tqdm

from mtgml.constants import MAX_BASICS, MAX_CARDS_IN_PACK, MAX_PICKED, MAX_SEEN_PACKS
from mtgml.utils.grid import interpolate

locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

with open('data/maps/card_to_int.json') as fp:
    card_to_int = json.load(fp)
with open('data/maps/int_to_card.json') as fp:
    int_to_card = json.load(fp)
with open('data/maps/old_int_to_card.json') as fp:
    old_int_to_card = json.load(fp)
old_int_to_new_int = [card_to_int[c["oracle_id"]] if c['oracle_id'] in card_to_int else -1 for c in old_int_to_card]
default_basic_ids = [
    "56719f6a-1a6c-4c0a-8d21-18f7d7350b68",
    "b2c6aa39-2d2a-459c-a555-fb48ba993373",
    "bc71ebf6-2056-41f7-be35-b2e5c34afa99",
    "b34bb2dc-c1af-4d77-b0b3-a0fb342a5fc6",
    "a3fb7228-e76b-4e96-a40e-20b5fed75685",
]
default_basics = tuple(card_to_int[c] for c in default_basic_ids)

BUFFER_SIZE = 1024 // 64


def pad(arr, desired_length, value=0):
    if isinstance(arr, tuple):
        if len(arr) < desired_length:
            arr = list(arr)
        else:
            return arr[:desired_length]
    return arr + [value for _ in range(desired_length - len(arr))]


def picks_from_draft(draft):
    if isinstance(draft, dict):
        basics = [x + 1 for x in draft.get('basics', default_basics)][:MAX_BASICS]
        if 'picks' in draft:
            seen = []
            seen_coords = []
            seen_coord_weights = []
            trashed = []
            value = None
            for pick in draft['picks']:
                picked_idx = pick.get('pickedIdx', pick.get('trashedIdx', None))
                if not (all(isinstance(x, int) for x in pick['cardsInPack'])
                        and all(isinstance(x, int) for x in pick['picked'])
                        and len(seen) < MAX_SEEN_PACKS
                        and picked_idx is not None and 0 <= picked_idx < len(pick['cardsInPack'])
                        and len(pick['picked']) <= MAX_PICKED):
                    break
                # if len(pick['cardsInPack']) <= 1: continue
                cards_in_pack = [x + 1 for x in pick['cardsInPack']]
                coords, coord_weights = interpolate(pick['pickNum'], pick['numPicks'],
                                                    pick['packNum'], pick['numPacks'])
                cards_in_pack[0], cards_in_pack[picked_idx] = cards_in_pack[picked_idx], cards_in_pack[0]
                cards_in_pack = cards_in_pack[:MAX_CARDS_IN_PACK]
                seen.append(cards_in_pack)
                seen_coords.append(coords)
                seen_coord_weights.append(coord_weights)
                trashed.append(0 if 'pickedIdx' in pick else 1)
                value = tuple(x + 1 for x in pick['picked'])
            if value is not None:
                yield (basics, value, seen, seen_coords, seen_coord_weights, trashed)


ALL_PICKS = tuple(0 for _ in range(MAX_SEEN_PACKS))


def picks_from_draft2(draft):
    if 'picks' in draft:
        BUFFER = np.zeros((len(int_to_card) + 1,), dtype=np.int16)
        seen = []
        seen_coords = []
        seen_coord_weights = []
        trashed = ALL_PICKS
        value = None
        for pick in draft['picks']:
            if not (all(isinstance(x, int) for x in pick['cardsInPack'])
                    and all(isinstance(x, int) for x in pick['picked'])
                    and len(seen) < MAX_SEEN_PACKS
                    and len(pick['picked']) <= MAX_PICKED):
                return
            cards_in_pack = [old_int_to_new_int[x] + 1 for x in pick['cardsInPack']]
            coords, coord_weights = interpolate(pick['pick'], pick['packSize'], pick['pack'],
                                                pick['packs'])
            chosen_card = old_int_to_new_int[pick['chosenCard']] + 1
            picked_idx = cards_in_pack.index(chosen_card)
            if picked_idx < 0:
                return
            cards_in_pack[0], cards_in_pack[picked_idx] = cards_in_pack[picked_idx], cards_in_pack[0]
            cards_in_pack = cards_in_pack[:MAX_CARDS_IN_PACK]
            seen.append(cards_in_pack[:MAX_CARDS_IN_PACK])
            seen_coords.append(coords)
            seen_coord_weights.append(coord_weights)
            value = tuple(old_int_to_new_int[x] + 1 for x in pick['picked'])
        if value is not None:
            yield (default_basics, value, seen, seen_coords, seen_coord_weights, ALL_PICKS)


DESTS = [0, 0, 0, 0, 0, 0, 0, 0, 1, 2]


def load_all_drafts(pool, *args):
    for drafts_dir, picks_gen in zip(args, (picks_from_draft, picks_from_draft2)):
        def load_drafts_file(drafts_file):
            try:
                with open(drafts_file, 'rb') as fp:
                    drafts = JsonSlicer(fp, (None,))

                    def gen():
                        for draft in tqdm(drafts, leave=False, dynamic_ncols=True, unit='draft', unit_scale=1,
                                          smoothing=0.001):
                            yield from picks_gen(draft)
                    return list(gen())
            except:
                logging.exception(f'Error in file {drafts_file}')
                return []
        for draft_dir in drafts_dir.split(';'):
            files = glob.glob(f'{draft_dir}/*.json')
            yield from tqdm((load_drafts_file(file) for file in files),
                            leave=False, dynamic_ncols=True, unit='file', unit_scale=1, total=len(files))


COLORS = {'W': 0, 'w': 0, 'U': 1, 'u': 1, 'B': 2, 'b': 2, 'R': 3, 'r': 3, 'G': 4, 'g': 4}


def colors_to_mask(colors):
    colors = set(COLORS[c] for c in colors if c in COLORS)
    result = np.zeros((5,), dtype=np.float32)
    result[list(colors)] = 1.0
    return result


def calculate_devotion(cost, cmc, color_identity, type_line):
    final_cost = np.zeros((5,), dtype=np.float32)
    if 'Land' in type_line:
        for color in color_identity:
            final_cost[COLORS[color]] = 1 / len(color_identity)
        return final_cost
    for symbol in cost:
        if '2' in symbol or 'p' in symbol:
            continue
        add = np.zeros((5,), dtype=np.float32)
        for c in symbol:
            if c in COLORS:
                add[COLORS[c]] += 1
        final_cost += add / max(add.sum(), 1)
    return final_cost * final_cost.sum() / max(cmc, 1)


DEVOTION = np.array([np.zeros((5,), dtype=np.float32)] + [calculate_devotion(c['parsed_cost'], c['cmc'], c['color_identity'], c['type']) for c in int_to_card])
COLOR_IDENTITY = np.array([np.zeros((5,), dtype=np.float32)] + [colors_to_mask(card['color_identity']) for card in int_to_card])
DEVOTION_AND_IDENTITY = DEVOTION + COLOR_IDENTITY


def calculate_riskiness(pack, pool):
    # total_devotions is BUFFER_SIZE, MAX_PICKED, 5
    total_devotions = np.cumsum(DEVOTION[pool], axis=-2)
    total_devotion = np.sum(total_devotions, axis=-1, keepdims=True)
    # pack_colors is BUFFER_SIZE, MAX_SEEN_PACKS, MAX_CARDS_IN_PACK, 5
    pack_colors = DEVOTION_AND_IDENTITY[pack]
    pack_used_colors = np.where((total_devotions > total_devotion / 5)[:, :, None] | (np.take(pack_colors, [0], axis=-2) > 0),
                                pack_colors, np.zeros_like(pack_colors))
    extra_colors = pack_colors.sum(axis=-1) - pack_used_colors.sum(axis=-1)
    pack_unused_colors = np.where((total_devotion + np.zeros_like(extra_colors)) > 5, extra_colors, np.zeros_like(extra_colors))
    return pack_unused_colors * 4 + 1


PREFIX = struct.Struct(f'{MAX_BASICS}H{MAX_PICKED}H{MAX_SEEN_PACKS * MAX_CARDS_IN_PACK}H{MAX_SEEN_PACKS * 4 * 2}B{MAX_SEEN_PACKS * 4}f{MAX_SEEN_PACKS * MAX_CARDS_IN_PACK}f{MAX_SEEN_PACKS}B')
POOL_ARRAY = np.zeros((BUFFER_SIZE, MAX_PICKED), dtype=np.int32)
PACK_ARRAY = np.zeros((BUFFER_SIZE, MAX_SEEN_PACKS, MAX_CARDS_IN_PACK), dtype=np.int32)
print(PREFIX.size)
sys.exit(0)


def write_picks(picks, output_file):
    for i, pick in enumerate(picks):
        POOL_ARRAY[i, :len(pick[1])] = pick[1]
        POOL_ARRAY[i, len(pick[1]):] = 0
        for j, pack in enumerate(pick[2]):
            PACK_ARRAY[i, j, :len(pack)] = pack
            PACK_ARRAY[i, j, len(pack):] = 0
        PACK_ARRAY[i, len(pick[2]):] = 0
    riskinesses = calculate_riskiness(PACK_ARRAY, POOL_ARRAY).astype(np.float32)
    for pick, riskiness in zip(picks, riskinesses.tolist()):
        write_pick(pick, riskiness, output_file)


def write_pick(pick, riskiness, output_file):
    basics, picked, seen, seen_coords, seen_coord_weights, trashed = pick
    seen = pad([x for pack in seen for x in pad(pack, MAX_CARDS_IN_PACK)], MAX_CARDS_IN_PACK * MAX_SEEN_PACKS)
    seen_coords = pad([y for pack in seen_coords for x in pack for y in x], MAX_SEEN_PACKS * 4 * 2)
    seen_coord_weights = pad([x for pack in seen_coord_weights for x in pack], MAX_SEEN_PACKS * 4, value=0.0)
    riskiness = [x for xs in riskiness for x in xs]
    args = []

    prefix = PREFIX.pack(*pad(basics, MAX_BASICS), *pad(picked, MAX_PICKED), *seen, *seen_coords, *seen_coord_weights,
                         *riskiness, *pad(trashed, MAX_PICKED))
    output_file.write(prefix)


def split_to(n, arr):
    return [arr[i * n:(i + 1) * n] for i in range(len(arr) // n)]


def read_pick(input_file, offset):
    coords = [[0 for _ in range(2)] for _ in range(4)]
    coord_weights = [0 for _ in range(4)]
    input_file.seek(offset)
    prefix_bytes = input_file.read(PREFIX.size)
    parsed = PREFIX.unpack(prefix_bytes)
    offset = 0
    cards_in_pack = parsed[:MAX_CARDS_IN_PACK]
    parsed = parsed[MAX_CARDS_IN_PACK:]
    basics = parsed[:MAX_BASICS]
    parsed = parsed[MAX_BASICS:]
    picked = parsed[:MAX_PICKED]
    parsed = parsed[MAX_PICKED:]
    seen = split_to(MAX_CARDS_IN_PACK, parsed[:MAX_CARDS_IN_PACK * MAX_SEEN_PACKS])
    parsed = parsed[MAX_CARDS_IN_PACK * MAX_SEEN_PACKS:]
    seen_coords = split_to(2, parsed[:MAX_SEEN_PACKS * 4 * 2])
    seen_coords = split_to(4, seen_coords)
    parsed = parsed[MAX_SEEN_PACKS * 4 * 2:]
    seen_coord_weights = split_to(4, parsed[:MAX_SEEN_PACKS * 4])
    parsed = parsed[MAX_SEEN_PACKS * 4:]
    coords = split_to(2, parsed[:4 * 2])
    parsed = parsed[4 * 2:]
    coord_weights = parsed[:4]
    parsed = parsed[4:]
    trashed = parsed[0]
    return cards_in_pack, basics, picked, seen, seen_coords, seen_coord_weights, coords, coord_weights, \
           trashed


def dump_picks(picks, input_file, dest_folder):
    context_count = len(picks)
    cards_in_pack = np.memmap(dest_folder/'cards_in_pack.npy', dtype=np.int16, mode='w+', shape=(context_count, MAX_CARDS_IN_PACK))
    basics = np.memmap(dest_folder/'basics.npy', dtype=np.int16, mode='w+', shape=(context_count, MAX_PICKED))
    picked = np.memmap(dest_folder/'picked.npy', dtype=np.int16, mode='w+', shape=(context_count, MAX_PICKED))
    seen = np.memmap(dest_folder/'seen.npy', dtype=np.int16, mode='w+', shape=(context_count, MAX_SEEN_PACKS, MAX_CARDS_IN_PACK))
    seen_coords = np.memmap(dest_folder/'seen_coords.npy', dtype=np.int8, mode='w+',
                            shape=(context_count, MAX_SEEN_PACKS, 4, 2))
    seen_coord_weights = np.memmap(dest_folder/'seen_coord_weights.npy', dtype=np.float32, mode='w+',
                                   shape=(context_count, MAX_SEEN_PACKS, 4))
    coords = np.memmap(dest_folder/'coords.npy', dtype=np.int8, mode='w+', shape=(context_count, 4, 2))
    coord_weights = np.memmap(dest_folder/'coord_weights.npy', dtype=np.float32, mode='w+', shape=(context_count, 4))
    y_idx = np.memmap(dest_folder/'y_idx.npy', dtype=np.int8, mode='w+', shape=(context_count,))
    for context_idx, offset in enumerate(tqdm(picks, leave=False, dynamic_ncols=True, unit='picks',
                                              unit_scale=1, smoothing=0.001)):
        pick = read_pick(input_file, offset)
        cards_in_pack[context_idx] = np.array(pick[0], dtype=np.int16)
        basics[context_idx] = np.array(pick[1], dtype=np.int16)
        picked[context_idx] = np.array(pick[2], dtype=np.int16)
        seen[context_idx] = np.array(pick[3], dtype=np.int16)
        seen_coords[context_idx] = np.array(pick[4], dtype=np.int8)
        seen_coord_weights[context_idx] = np.array(pick[5], dtype=np.float32)
        coords[context_idx] = np.array(pick[6], dtype=np.int8)
        coord_weights[context_idx] = np.array(pick[7], dtype=np.float32)
        y_idx[context_idx] = np.array(pick[8], dtype=np.int8)
    with open(dest_folder / 'counts.json', 'w') as count_file:
        json.dump({"contexts": context_count}, count_file)
    print(f'{dest_folder} has {context_count:n} picks.')
    cctx = zstd.ZstdCompressor(level=10, threads=-1)
    for name, arr in (('picked', picked), ('seen', seen),
                      ('seen_coords', coords), ('seen_coord_weights', coord_weights),
                      ('coords', coords), ('coord_weights', coord_weights),
                      ('y_idx', y_idx), ('cards_in_pack', cards_in_pack)):
        with open(dest_folder / f'{name}.npy.zstd', 'wb') as fh:
            with cctx.stream_writer(fh) as compressor:
                np.save(compressor, arr, allow_pickle=False)
        print(f'Saved {name} with zstd.')
    return context_count


if __name__ == '__main__':
    train_filename = Path('data/train_picks.bin')
    validation_filename = Path('data/validation_picks.bin')
    evaluation_filename = Path('data/evaluation_picks.bin')
    num_picks = 0
    with open(train_filename, 'wb') as train_file, open(validation_filename, 'wb') as validation_file, \
          open(evaluation_filename, 'wb') as evaluation_file, \
          multiprocessing.pool.ThreadPool(processes=1) as read_pool, \
          multiprocessing.pool.ThreadPool(processes=1) as write_pool:
        output_files = [train_file, validation_file, evaluation_file]
        for picks in load_all_drafts(read_pool, *sys.argv[1:]):
            for i in range(len(picks) // BUFFER_SIZE):
                rand_val = random.randint(0, 9)
                dest = DESTS[rand_val]
                write_pool.apply(write_picks, (picks[i * BUFFER_SIZE:(i + 1) * BUFFER_SIZE], output_files[dest]))
                num_picks += BUFFER_SIZE
        read_pool.close()
        write_pool.close()
        read_pool.join()
        write_pool.join()
    print(f'There were {num_picks} total picks')
