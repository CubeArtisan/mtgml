FROM python:3.10-bullseye

WORKDIR /mtgml

COPY serving_requirements.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p mtgml/server mtgml/utils ml_files/tflite
COPY mtgml/server/*.py mtgml/server/
COPY mtgml/constants.py mtgml/constants.py
COPY mtgml/utils/grid.py mtgml/utils/grid.py
COPY ml_files/tflite/ ml_files/tflite/

ENTRYPOINT gunicorn -w 2 --access-logfile - -b 0.0.0.0:8000 --timeout 300 mtgml.server:app
