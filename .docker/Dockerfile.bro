FROM python:3.9-bullseye

WORKDIR /mtgml

COPY serving_requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p mtgml/server mtgml/utils ml_files/tflite
COPY mtgml/server/*.py mtgml/server/
COPY mtgml/constants.py mtgml/constants.py
COPY mtgml/utils/grid.py mtgml/utils/grid.py
COPY ml_files/tflite/bro_int_to_oracle_id.json ml_files/tflite/
COPY ml_files/tflite/bro_model.tflite ml_files/tflite/
COPY ml_files/tflite/bro_original_to_new_index.json ml_files/tflite/

ENTRYPOINT gunicorn -w 8 --access-logfile - -b 0.0.0.0:8000 --timeout 300 mtgml.server:app