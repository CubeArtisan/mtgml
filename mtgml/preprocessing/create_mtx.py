import json
from pathlib import Path

import numpy as np

import mtgml.preprocessing.utils as utils


def is_valid_deck(deck):
    # return True
    return len(deck['side']) > 0


if __name__ == '__main__':
    data = Path('data')
    maps = data / 'maps'
    int_to_card_filepath = maps / 'int_to_card.json'
    decks_folder = data / 'decks'
    print('Loading Deck Data.')
    num_decks = 0
    with open(int_to_card_filepath, 'rb') as int_to_card_file:
        int_to_card = json.load(int_to_card_file)
    card_to_int = {v: i for i, v in enumerate(int_to_card)}
    num_cards = len(int_to_card)

    adj_mtx = utils.build_mtx(decks_folder, len(int_to_card),
                              card_to_int, validation_func=is_valid_deck,
                              soft_validation=0)


    np.save(data / 'adj_mtx.npy', adj_mtx)

    with open(maps / 'int_to_card.json', 'w') as out_lookup:
        json.dump(int_to_card,  out_lookup)
