import glob
import json
import locale
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
with open('data/maps/old_int_to_card.json') as fp:
    old_int_to_card = json.load(fp)
old_int_to_new_int = [card_to_int[c["oracle_id"]] for c in old_int_to_card]
default_basic_ids = [
    "56719f6a-1a6c-4c0a-8d21-18f7d7350b68",
    "b2c6aa39-2d2a-459c-a555-fb48ba993373",
    "bc71ebf6-2056-41f7-be35-b2e5c34afa99",
    "b34bb2dc-c1af-4d77-b0b3-a0fb342a5fc6",
    "a3fb7228-e76b-4e96-a40e-20b5fed75685",
]
default_basics = [card_to_int[c] for c in default_basic_ids]


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
            for pick in draft['picks']:
                if not (all(isinstance(x, int) for x in pick['cardsInPack']) and
                        all(isinstance(x, int) for x in pick['picked'])):
                    break
                picked_idx = pick.get('pickedIdx', pick.get('trashedIdx', None))
                cards_in_pack = [x + 1 for x in pick['cardsInPack']]
                coords, coord_weights = interpolate(pick['pickNum'], pick['numPicks'],
                                                    pick['packNum'], pick['numPacks'])
                seen.append(cards_in_pack[:MAX_CARDS_IN_PACK])
                seen_coords.append(coords)
                seen_coord_weights.append(coord_weights)
                if picked_idx is not None and 0 <= picked_idx < len(cards_in_pack):
                    chosen = cards_in_pack[picked_idx]
                    cards_in_pack = set(cards_in_pack)
                    cards_in_pack.remove(chosen)
                    cards_in_pack = [chosen, *cards_in_pack]
                    picked = [x + 1 for x in pick['picked']]
                    if len(picked) <= MAX_PICKED and len(seen) <= MAX_SEEN_PACKS:
                        trashed = 0 if 'pickedIdx' in pick else 1
                        yield (cards_in_pack[:MAX_CARDS_IN_PACK], basics, picked, tuple(seen),
                               tuple(seen_coords), tuple(seen_coord_weights), coords, coord_weights,
                               trashed)


def picks_from_draft2(draft):
    if 'picks' in draft:
        seen = []
        seen_coords = []
        seen_coord_weights = []
        for pick in draft['picks']:
            if not (all(isinstance(x, int) for x in pick['cardsInPack']) and
                    all(isinstance(x, int) for x in pick['picked'])):
                break
            cards_in_pack = list(old_int_to_new_int[x] + 1 for x in pick['cardsInPack'])
            coords, coord_weights = interpolate(pick['pick'], pick['packSize'], pick['pack'],
                                                pick['packs'])
            seen.append(cards_in_pack[:MAX_CARDS_IN_PACK])
            seen_coords.append(coords)
            seen_coord_weights.append(coord_weights)
            chosen_card = old_int_to_new_int[pick['chosenCard']] + 1
            if chosen_card not in cards_in_pack:
                continue
            cards_in_pack = set(cards_in_pack)
            cards_in_pack.remove(chosen_card)
            cards_in_pack = [chosen_card, *cards_in_pack]
            picked = [old_int_to_new_int[x] + 1 for x in pick['picked']]
            if len(picked) <= MAX_PICKED and len(seen) <= MAX_SEEN_PACKS:
                yield (cards_in_pack[:MAX_CARDS_IN_PACK], default_basics, picked, tuple(seen),
                       tuple(seen_coords), tuple(seen_coord_weights), coords, coord_weights, 0)


DESTS = [0, 0, 0, 0, 0, 0, 0, 0, 1, 2]

def load_all_drafts(*args):
    num_drafts = 0
    for drafts_dir, picks_gen in zip(args, (picks_from_draft, picks_from_draft2)):
        for draft_dir in drafts_dir.split(';'):
            for drafts_file in tqdm(glob.glob(f'{draft_dir}/*.json'), leave=False, dynamic_ncols=True,
                                    unit='file', unit_scale=1):
                with open(drafts_file, 'rb') as fp:
                    drafts = JsonSlicer(fp, (None,))
                    for draft in tqdm(drafts, leave=False, dynamic_ncols=True, unit='draft', unit_scale=1,
                                      smoothing=0.001, initial=num_drafts):
                        num_drafts += 1
                        rand_val = random.randint(0, 9)
                        dest = DESTS[rand_val]
                        for pick in picks_gen(draft):
                            yield (dest, pick)
    print(f'Total drafts {num_drafts:n}')

PREFIX = struct.Struct(f'{MAX_CARDS_IN_PACK}H{MAX_BASICS}H{MAX_PICKED}H{MAX_SEEN_PACKS * MAX_CARDS_IN_PACK}H{MAX_SEEN_PACKS * 4 * 2}B{MAX_SEEN_PACKS * 4}f8B4fB3x')


def write_pick(pick, output_file):
    cards_in_pack, basics, picked, seen, seen_coords, seen_coord_weights, coords, coord_weights, trashed = pick
    seen = pad([x for pack in seen for x in pad(pack, MAX_CARDS_IN_PACK)], MAX_CARDS_IN_PACK * MAX_SEEN_PACKS)
    seen_coords = pad([y for pack in seen_coords for x in pack for y in x], MAX_SEEN_PACKS * 4 * 2)
    seen_coord_weights = pad([x for pack in seen_coord_weights for x in pack], MAX_SEEN_PACKS * 4)
    coords = [x for coord in coords for x in coord]
    prefix = PREFIX.pack(*pad(cards_in_pack, MAX_CARDS_IN_PACK), *pad(basics, MAX_BASICS),
                         *pad(picked, MAX_PICKED), *seen, *seen_coords, *seen_coord_weights, *coords,
                         *coord_weights, trashed)
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
    train_filename = Path('data/train_picks.json')
    validation_filename = Path('data/validation_picks.json')
    evaluation_filename = Path('data/evaluation_picks.json')
    with open(train_filename, 'wb') as train_file, open(validation_filename, 'wb') as validation_file, \
          open(evaluation_filename, 'wb') as evaluation_file:
        output_files = [train_file, validation_file, evaluation_file]
        for dest, pick in load_all_drafts(*sys.argv[1:]):
            write_pick(pick, output_files[dest])
