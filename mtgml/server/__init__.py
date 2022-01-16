import json
import logging
from pathlib import Path

import tensorflow as tf
from flask import Flask, request

from mtgml.server.cube_recommender import get_cube_recomendations
from mtgml.server.draftbots import get_draft_scores


app = Flask(__name__)

if __name__ != "__main__":
    gunicorn_error_logger = logging.getLogger("gunicorn.error")
    app.logger.handlers.extend(gunicorn_error_logger.handlers)
    app.logger.setLevel(logging.INFO)

MODEL_PATH = Path('ml_files/current')
MODEL = None
with open(MODEL_PATH / 'int_to_card.json', 'rb') as map_file:
    int_to_card = json.load(map_file)
CARD_TO_INT = {v: k for k, v in enumerate(int_to_card)}
INT_TO_CARD = {int(k): v for k, v in enumerate(int_to_card)}


def get_model():
    global MODEL
    if MODEL is None:
        MODEL = tf.keras.models.load_model(MODEL_PATH)
    return MODEL


# This is a POST request since it needs to take in a cube object
@app.route("/cube", methods=['POST'])
def cube_recommendations():
    num_recs = request.args.get("num_recs", 30_000)
    json_data = request.get_json()
    if json_data is None or "cube" not in json_data:
        error = "Must have a valid cube at the cube key in the json body."
        app.logger.error(error)
        return { "success": False, "error": error }, 400
    cube = json_data.get("cube")
    model = get_model()
    try:
        results = get_cube_recomendations(cube=cube, model=model, card_to_int=CARD_TO_INT, num_recs=num_recs)
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


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=8000, threaded=True)
