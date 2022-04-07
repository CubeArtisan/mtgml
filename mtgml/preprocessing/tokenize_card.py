import re
import sys

from mtgml.preprocessing.tokens import tokens, extra_productions, referenced_card_names, extra_card_names, regexes


class TokenTrie:
    def __init__(self, words):
        self.inhabited = '' in words
        first_chars = {c[0] for c in words if len(c) > 0}
        self.lookup = {c: TokenTrie([w[1:] for w in words if len(w) > 0 and w[0] == c]) for c in first_chars}

    def __getitem__(self, query):
        if query != '' and query[0] in self.lookup:
            recursed = self.lookup[query[0]][query[1:]]
            if recursed is not None:
                return query[0] + recursed
        if self.inhabited:
            return ''
        else:
            return None

token_to_idx = {v: i for i, v in enumerate(tokens)}
productions = {
    p[0]: p[1]
    for p in
       [(t, (t,)) for t in tokens]
        + list(extra_productions)
        + [(t, ('~~',)) for t in referenced_card_names]
        + [(t, ('~',)) for t in extra_card_names]
}
usage_table = {k: 0 for k in productions.keys()}
max_len = max(len(k) for k in productions.keys())
if sys.getrecursionlimit() < 10 * max_len:
    sys.setrecursionlimit(10 * max_len)
production_trie = TokenTrie(productions.keys())


def tokenize_string(txt):
    tokenized = []
    txt = txt.lower()
    og_text = txt
    for match, replacement in regexes:
        txt = re.sub(match, replacement, txt)
    processed_text = txt
    while len(txt) > 0:
        prefix = production_trie[txt]
        if prefix is None:
            print(og_text, '\n')
            print(processed_text, '\n')
            print(txt)
            raise Exception('Could not produce txt from tokens.')
        else:
            usage_table[prefix] += 1
            for token in productions[prefix]:
                if token != prefix:
                    usage_table[token] += 1
                tokenized.append(token_to_idx[token])
            txt = txt[len(prefix):]
    return tokenized


def tokenize_card(card):
    card_string = '[[card]]'
    if 'power' in card:
        card_string += f' [[power]] {card["power"]}'
    if 'toughness' in card:
        card_string += f' [[toughness]] {card["toughness"]}'
    if 'loyalty' in card:
        card_string += f' [[loyalty]] {card["loyalty"]}'
    filtered_cost = [c for c in card['parsed_cost'] if len(c) > 0]
    if len(filtered_cost) > 0:
        card_string += f' [[cost]] {" ".join("{" + c + "}" for c in filtered_cost)}'
    card_string += f' [[type]] {card["type"]}'
    oracle = card["oracle_text"]
    name = card["name"]
    oracle = oracle.replace(name, '~')
    if ',' in name:
        name = name.split(',')[0]
        oracle = re.sub(r'\b' + name + r'\b', '~', oracle)
    oracle = re.sub(r"[^•]+ — ", '', oracle) # remove ability words we want to focus on mechanics.
    card_string += f' [[oracle]] {oracle}'
    return tokenize_string(card_string)


def untokenize(tkns):
    return ' '.join(tokens[tkn] for tkn in tkns)


if __name__ == '__main__':
    import json
    with open('data/maps/int_to_card.json') as fp:
        int_to_card = json.load(fp)
    tokenized = [tokenize_card(c) for i, c in enumerate(int_to_card)]
