import re

tokens = {'[nil]': 0, '[mask]': 1}
def getToken(tkn):
    tkn = tkn.lower()
    if tkn not in tokens:
        tokens[tkn] = len(tokens)
    return tokens[tkn]


def tokenizeColor(color):
    return '{' + color.replace('-', '/') + '}'


NUMBERS = {
    'one': '1',
    'two': '2',
    'three': '3',
    'four': '4',
    'five': '5',
    'six': '6',
    'seven': '7',
    'eight': '8',
    'nine': '9',
    'ten': '10',
    'eleven': '11',
    'twelve': '12',
    'thirteen': '13',
    'fourteen': '14',
    'fifteen': '15',
    'sixteen': '16',
    'seventeen': '17',
    'eighteen': '18',
    'nineteen': '19',
    'twenty': '20',
    'twenty-six': '26',
    'forty': '40',

    # 'ninety-9': '99',
}

def processOracleText(txt, name):
    txt = txt.replace(name, '~')
    if ',' in name:
        name = name.split(',')[0]
        txt = re.sub(r'\b' + name + r'\b', '~', txt)
    txt = txt.lower()
    txt = re.sub(r'(\{[^}]+\})', r' \1 ', txt) # split symbols apart
    txt = re.sub(r'([0-9x])\/([0-9+-x])', r'\1 / \2', txt) # split +x/+1 apart

    # txt = re.sub(r'([0-9x½])\/([0-9+-x])', r'\1 / \2', txt) # split +x/+1 apart

    txt = re.sub(r'\([^)]+\)', '', txt) # remove reminder text
    for word, num in NUMBERS.items():
        txt = re.sub(r'\b' + word + r'\b', num, txt)
    txt = re.sub(r"\b'", ' " ', txt) # split signle quotes but not contractions.

    txt = re.sub(r"^.+ — ", '', txt, flags=re.MULTILINE)
    txt = re.sub(r"([^a-z])'", r'\1 " ', txt) # split signle quotes but not contractions.
    txt = re.sub(r'^-([^0-9])', r'\1', txt)
    txt = re.sub(r'([0-9])-([0-9])', r'\1 to \2', txt)
    txt = txt.replace('non-', 'non ')
    txt = re.sub(r'\bnon([^e])', r'non \1', txt)
    txt = txt.replace('this spell', ' ~ ')
    txt = txt.replace('®', '')

    txt = txt.replace(':', ' : ')
    txt = txt.replace('.', ' . ')
    txt = txt.replace(',', ' , ')
    txt = txt.replace('+', ' + ')
    txt = txt.replace(';', ' ; ')
    txt = txt.replace('"', ' " ')
    txt = txt.replace('?', ' ? ')
    txt = txt.replace('!', ' ! ')
    txt = txt.replace("'s", " 's ")
    txt = txt.replace("’s", " 's ")

    txt = txt.replace("s'", "s 's ")

    txt = txt.replace('+', ' + ')
    txt = txt.replace('[', ' [ ')
    txt = txt.replace(']', ' ] ')
    txt = txt.replace('\n', ' [CRLF] ')
    txt = re.sub(r' +', ' ', txt) # remove repeated spaces.
    return txt


def tokenizeString(txt):
    return [getToken(s) for s in txt.split(' ')]


def tokenizeCard(card):
    result = [getToken('[CARD]')]
    filtered_cost = [c for c in card['parsed_cost'] if len(c) > 0]
    if len(filtered_cost) > 0:
        result.append(getToken('[COST]'))
        result += [getToken(tokenizeColor(c)) for c in filtered_cost]
    result.append(getToken('[TYPE]'))
    result += tokenizeString(card['type'])
    oracle_text = processOracleText(card['oracle_text'], card['name'])
    if len(oracle_text) > 0:
        result.append(getToken('[ORACLE]'))
        result += tokenizeString(oracle_text)
    if 'power' in card:
        result += [getToken('[POWER]'), getToken(card['power'])]
    if 'toughness' in card:
        result += [getToken('[TOUGNESS]'), getToken(card['toughness'])]
    if 'loyalty' in card:
        result += [getToken('[LOYALTY]'), getToken(card['loyalty'])]
    return result


def untokenize(tkns):
    inverse_tokens = {v: k for k, v in tokens.items()}
    return ' '.join(inverse_tokens[tkn] for tkn in tkns)

if __name__ == '__main__':
    import json
    with open('data/maps/int_to_card.json') as fp:
        int_to_card = json.load(fp)
    tokenized = [tokenizeCard(c) for c in int_to_card]
