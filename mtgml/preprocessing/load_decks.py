import glob
import json
import locale
import random
import struct
import sys
from pathlib import Path

from jsonslicer import JsonSlicer
from tqdm.auto import tqdm

from mtgml.constants import MAX_DECK_SIZE, MAX_SIDEBOARD_SIZE

locale.setlocale(locale.LC_ALL, 'en_US.UTF-8')

with open('data/maps/card_to_int.json') as fp:
    card_to_int = json.load(fp)
with open('data/maps/int_to_card.json') as fp:
    int_to_card = json.load(fp)
name_to_int = {card['name_lower']: i for i, card in enumerate(int_to_card)}


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
    for deck_dir in deck_dirs.split(';'):
        for decks_file in tqdm(glob.glob(f'{deck_dir}/*.json'), leave=False, dynamic_ncols=True,
                                unit='file', unit_scale=1):
            with open(decks_file, 'rb') as fp:
                decks = JsonSlicer(fp, (None,))
                for deck in tqdm(decks, leave=False, dynamic_ncols=True, unit='deck', unit_scale=1,
                                  smoothing=0.001, initial=num_decks):
                    if len(deck['main']) >= 22 and all(isinstance(x, int) for x in deck['main']) \
                       and len(deck['side']) > 0 and all(isinstance(x, int) for x in deck['side']):
                        num_decks += 1
                        rand_val = random.randint(0, 9)
                        dest = DESTS[rand_val]
                        yield (dest, (tuple(x + 1 for x in deck['main'][:MAX_DECK_SIZE]),
                                      tuple(x + 1 for x in deck['side'][:MAX_SIDEBOARD_SIZE])))
    print(f'Total decks {num_decks:n}')


def load_all_old_decks(deck_dirs):
    num_decks = 0
    for deck_dir in deck_dirs.split(';'):
        for decks_file in tqdm(glob.glob(f'{deck_dir}/*.json'), leave=False, dynamic_ncols=True,
                                unit='file', unit_scale=1):
            with open(decks_file, 'rb') as fp:
                decks = JsonSlicer(fp, (None,))
                for deck in tqdm(decks, leave=False, dynamic_ncols=True, unit='deck', unit_scale=1,
                                  smoothing=0.001, initial=num_decks):
                    if len(deck['main']) >= 22 and all(x in name_to_int for x in deck['main']) \
                       and len(deck['side']) > 0 and all(x in name_to_int for x in deck['side']):
                        num_decks += 1
                        rand_val = random.randint(0, 9)
                        dest = DESTS[rand_val]
                        yield (dest, (tuple(name_to_int[x] + 1 for x in deck['main'][:MAX_DECK_SIZE]),
                                      tuple(name_to_int[x] + 1 for x in deck['side'][:MAX_SIDEBOARD_SIZE])))
    print(f'Total decks {num_decks:n}')


PREFIX = struct.Struct(f'{MAX_DECK_SIZE}H{MAX_SIDEBOARD_SIZE}H')


def write_deck(deck, output_file):
    prefix = PREFIX.pack(*pad(deck[0], MAX_DECK_SIZE), *pad(deck[1], MAX_SIDEBOARD_SIZE))
    output_file.write(prefix)


if __name__ == '__main__':
    train_filename = Path('data/train_decks.bin')
    validation_filename = Path('data/validation_decks.bin')
    evaluation_filename = Path('data/evaluation_decks.bin')
    with open(train_filename, 'wb') as train_file, open(validation_filename, 'wb') as validation_file, \
          open(evaluation_filename, 'wb') as evaluation_file:
        output_files = [train_file, validation_file, evaluation_file]
        for dest, deck in load_all_decks(sys.argv[1]):
            write_deck(deck, output_files[dest])
        for dest, deck in load_all_old_decks(sys.argv[2]):
            write_deck(deck, output_files[dest])
