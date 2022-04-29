import sys

import numpy as np
import numpy.ma as ma
import tensorflow as tf
import yaml

from mtgml.constants import MAX_BASICS, MAX_CARDS_IN_PACK, MAX_PICKED, MAX_SEEN_PACKS
from mtgml.utils.grid import interpolate
from mtgml.config.hyper_config import HyperConfig
from mtgml.models.draftbots import DraftBot

MODEL: DraftBot = None
INT_TO_CARD = []
CARD_TO_INT: dict = {}
LOGGER = None


def initialize_draftbots(MODEL_PATH, EMBEDDINGS, int_to_card, card_to_int, logger):
    global MODEL, INT_TO_CARD, CARD_TO_INT, LOGGER
    INT_TO_CARD = int_to_card
    CARD_TO_INT = card_to_int
    LOGGER = logger
    with open(MODEL_PATH/'draftbot_hyper_config.yaml', 'r') as config_file:
        data = yaml.load(config_file, yaml.Loader)
    hyper_config = HyperConfig(layer_type=DraftBot, data=data, fixed={
        'num_cards': len(INT_TO_CARD) + 1,
        'card_embeddings': EMBEDDINGS,
    })
    tf.config.optimizer.set_jit(bool(hyper_config.get_bool('use_xla', default=False, help='Whether to use xla to speed up calculations.')))
    MODEL = hyper_config.build(name='CombinedModel', dynamic=False)
    if MODEL is None:
        logger.error('Failed to load model')
        sys.exit(1)
    MODEL.compile()
    latest = MODEL_PATH / 'draftbot_model'
    MODEL.load_weights(latest).expect_partial()
    MODEL.build(())

def get_draft_scores(drafter_state):
    cards_in_pack = np.zeros((1, MAX_CARDS_IN_PACK), dtype=np.int32)
    original_cards_idx = []
    idx = 0
    for i, card_id in enumerate(drafter_state['cardsInPack']):
        if card_id in CARD_TO_INT and idx < MAX_CARDS_IN_PACK:
            cards_in_pack[0][idx] = CARD_TO_INT[card_id] + 1
            original_cards_idx.append(i)
            idx += 1
    basics = np.zeros((1, MAX_BASICS), dtype=np.int32)
    idx = 0
    for card_id in drafter_state['basics']:
        if card_id in CARD_TO_INT and idx < MAX_BASICS:
            basics[0][idx] = CARD_TO_INT[card_id] + 1
            idx += 1
    picked = np.zeros((1, MAX_PICKED), dtype=np.int32)
    idx = 0
    for card_id in drafter_state['picked']:
        if card_id in CARD_TO_INT and idx < MAX_PICKED:
            picked[0][idx] = CARD_TO_INT[card_id] + 1
            idx += 1
    seen_packs = np.zeros((1, MAX_SEEN_PACKS, MAX_CARDS_IN_PACK), dtype=np.int32)
    seen_coords = np.zeros((1, MAX_SEEN_PACKS, 4, 2), dtype=np.int32)
    seen_weights = np.zeros((1, MAX_SEEN_PACKS, 4), dtype=np.float32)
    idx_pack = 0
    for pack in drafter_state['seen']:
        if idx_pack >= MAX_SEEN_PACKS: break
        idx = 0
        seen_coords[0][idx_pack], seen_weights[0][idx_pack] = interpolate(pack['pickNum'], pack['numPicks'],
                                                                          pack['packNum'], drafter_state['numPacks'])
        for card_id in pack['pack']:
            if card_id in CARD_TO_INT and idx < MAX_CARDS_IN_PACK:
                seen_packs[0][idx_pack][idx] = CARD_TO_INT[card_id] + 1
                idx += 1
        idx_pack += 1
    coords, weights = interpolate(drafter_state['pickNum'], drafter_state['numPicks'], drafter_state['packNum'],
                                  drafter_state['numPacks'])
    assert MODEL is not None
    oracle_scores, oracle_weights = MODEL((cards_in_pack, basics, picked, seen_packs, seen_coords, seen_weights,
                                            [coords], [weights],), training=False)
    oracle_scores = oracle_scores.numpy()[0]
    oracle_scores = ma.masked_array(oracle_scores, mask=np.broadcast_to(np.expand_dims(cards_in_pack[0] == 0, -1), oracle_scores.shape))
    oracle_scores_min = np.amin(oracle_scores, axis=0)
    oracle_scores = oracle_scores - oracle_scores_min
    oracle_max_scores = np.max(oracle_scores, axis=0) / 10
    oracle_max_scores = np.where(oracle_max_scores > 0, oracle_max_scores, np.ones_like(oracle_max_scores))
    oracle_scores = oracle_scores / oracle_max_scores
    oracle_weights = oracle_weights.numpy()[0] * oracle_max_scores
    oracle_weights = oracle_weights / oracle_weights.sum()
    scores = (oracle_weights * oracle_scores).sum(axis=1)
    oracles = [[dict(metadata) | { 'weight': float(weight), 'score': float(score) }
                for metadata, weight, score
                in zip(MODEL.sublayer_metadata, oracle_weights, card_scores)]
               for card_scores in oracle_scores]
    returned_scores = [0.0 for _ in drafter_state["cardsInPack"]]
    returned_oracles = [[metadata | {'weight': float(weight), 'score': 0}
                         for metadata, weight in zip(MODEL.sublayer_metadata, oracle_weights)]
                        for _ in drafter_state['cardsInPack']]
    for i, score, oracle in zip(original_cards_idx, scores, oracles):
        returned_scores[i] = float(score)
        returned_oracles[i] = oracle
    return {
        "scores": returned_scores,
        "oracles": returned_oracles,
        "success": True,
    }
