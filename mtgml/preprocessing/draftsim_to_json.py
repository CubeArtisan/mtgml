import collections
import csv
import itertools
import json
import sys
from pathlib import Path

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


def split_not_follow(s, split_on=',', not_preceding='_'):
    last_char = 0
    for i, c in enumerate(s):
        if c == split_on and (i + 1 >= len(s) or s[i + 1] != not_preceding):
            yield s[last_char:i]
            last_char = i + 1
    yield s[last_char:]


def to_card_index(name, name_to_int):
    for basic in ['Plains', 'Island', 'Swamp', 'Mountain', 'Forest']:
        if name.startswith(basic):
            name = basic
    return name_to_int[name]


def reconstruct_packs(picks):
    packs = tuple(tuple([None for _ in range(15)] for _ in range(3)) for _ in picks)
    for k in range(3):
        for j in range(15):
            offset = j if k % 2 == 0 else -j
            for i in range(len(picks)):
                packs[i][k][j] = picks[(i + offset) % len(picks)][15 * k + j]
    counter = 0
    for round in packs:
        for pack in round:
            assert None not in pack
    return packs


def reconstruct_states(packs, picks):
    states_per_player = tuple(tuple({'numPicks': 15, 'numPacks': 3} for _ in range(45)) for _ in picks)
    seen_per_player = [[] for _ in picks]
    for pack_num in range(3):
        for pick_num in range(15):
            # This is the opposite of the above because it goes from player to pack not pack to player.
            offset = pick_num if pack_num % 2 == 1 else -pick_num
            pick_index = 15 * pack_num + pick_num
            for player_index, player_picks in enumerate(picks):
                pack = packs[(player_index + offset) % len(packs)][pack_num]
                seen_per_player[player_index] += pack
                picked = player_picks[pick_index]
                assert picked in pack
                picked_idx = pack.index(picked)
                states_per_player[player_index][pick_index]['seen'] = tuple(seen_per_player[player_index])
                states_per_player[player_index][pick_index]['picked'] = tuple(player_picks[:pick_index])
                states_per_player[player_index][pick_index]['cardsInPack'] = tuple(pack)
                states_per_player[player_index][pick_index]['pickedIdx'] = picked_idx
                states_per_player[player_index][pick_index]['pickNum'] = pick_num
                states_per_player[player_index][pick_index]['packNum'] = pack_num
                del pack[picked_idx]
    return [{ "picks": player_states, "basics": []} for player_states in states_per_player]


def process_file(name_to_int, filename):
    drafter_states = []
    with open(filename, 'r', newline='') as fp:
        num_lines = sum(1 for _ in fp)
    with open(filename, 'r', newline='') as fp:
        reader = csv.reader(fp)
        for line in tqdm(reader, total=num_lines, dynamic_ncols=True, unit='draft', unit_scale=True,
                         smoothing=0.01):
            picks = tuple(tuple(to_card_index(name, name_to_int) for name in split_not_follow(player))
                          for player in line[2:])
            packs = reconstruct_packs(picks)
            yield from reconstruct_states(packs, picks)


if __name__ == '__main__':
    with open(Path(sys.argv[1]).parent/'int_to_card.json') as fp:
        int_to_card = json.load(fp)
    name_to_int = {c['name'].replace(' ', '_'): i for i, c in enumerate(int_to_card)}
    state_iter = process_file(name_to_int, sys.argv[1])
    with open(sys.argv[2], 'w') as fp:
        json.dump(state_iter, fp, cls=IterEncoder)
