import json
import logging
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
import yaml
from flask import Flask, request

from mtgml.config.hyper_config import HyperConfig
from mtgml.models.combined_model import CombinedModel
from mtgml.server.cube_recommender import get_cube_recomendations
from mtgml.server.draftbots import get_draft_scores


app = Flask(__name__)

if __name__ != "__main__":
    gunicorn_error_logger = logging.getLogger("gunicorn.error")
    app.logger.handlers.extend(gunicorn_error_logger.handlers)
    app.logger.setLevel(logging.INFO)

MODEL_PATH = Path('ml_files/latest')
MODEL = None
with open(MODEL_PATH / 'int_to_card.json', 'rb') as map_file:
    int_to_card = json.load(map_file)
CARD_TO_INT = {v['oracle_id']: k for k, v in enumerate(int_to_card)}
INT_TO_CARD = {int(k): v for k, v in enumerate(int_to_card)}


def get_model():
    global MODEL
    if MODEL is None:
        tf.keras.utils.set_random_seed(1)
        tf.config.experimental.enable_op_determinism()
        with open(MODEL_PATH/'hyper_config.yaml', 'r') as config_file:
            data = yaml.load(config_file, yaml.Loader)
        hyper_config = HyperConfig(layer_type=CombinedModel, data=data, fixed={
            'num_cards': len(INT_TO_CARD) + 1,
            'DraftBots': { 'card_weights': np.ones((len(INT_TO_CARD) + 1,)) },
        })
        # tf.config.optimizer.set_jit(bool(hyper_config.get_bool('use_xla', default=False, help='Whether to use xla to speed up calculations.')))
        MODEL = hyper_config.build(name='CombinedModel', dynamic=False)
        if MODEL is None:
            app.logger.error('Failed to load model')
            sys.exit(1)
        MODEL.compile()
        latest = tf.train.latest_checkpoint(MODEL_PATH)
        MODEL.load_weights(latest)
        MODEL.build(())
    return MODEL


# This is a POST request since it needs to take in a cube object
@app.route("/cube", methods=['POST'])
def cube_recommendations():
    num_recs = int(request.args.get("num_recs", 100))
    json_data = request.get_json()
    if json_data is None or "cube" not in json_data:
        error = "Must have a valid cube at the cube key in the json body."
        app.logger.error(error)
        return { "success": False, "error": error }, 400
    cube = json_data.get("cube")
    model = get_model()
    try:
        results = get_cube_recomendations(cube=cube, model=model, card_to_int=CARD_TO_INT, int_to_card=INT_TO_CARD, num_recs=num_recs)
    except Exception as e:
        app.logger.error(e)
        raise
    return results


# This is a POST request since it needs to take in a cube object
@app.route("/draft", methods=['POST'])
def draft_recommendations():
    json_data = request.get_json()
    if json_data is None or "drafterState" not in json_data:
        error = "Must have a valid drafter state at the drafterState key in the json body."
        app.logger.error(error)
        return { "success": False, "error": error }, 400
    drafter_state = json_data.get("drafterState")
    model = get_model()
    try:
        results = get_draft_scores(drafter_state, model=model, card_to_int=CARD_TO_INT)
    except Exception as e:
        app.logger.error(e)
        raise
    return results

@app.route('/version', methods=['GET'])
def get_version():
    return { 'version': os.environ.get('MTGML_VERSION', 'alpha'), 'success': True }

@app.after_request
def after_request(response):
    response.headers.add('Access-Control-Allow-Origin',  "*")
    response.headers.add('Access-Control-Allow-Headers', "*")
    response.headers.add('Access-Control-Allow-Methods', "*")
    return response

if __name__ == "__main__":
    get_model()
    app.run(host="0.0.0.0", port=8000, threaded=True)
