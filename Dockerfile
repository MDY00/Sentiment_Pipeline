FROM python:3.10

WORKDIR /app

COPY requirements.txt .

RUN pip --no-cache-dir install -r requirements.txt
RUN pip install -U pip setuptools wheel && pip install -U spacy && python -m spacy download en_core_web_sm

ENV PYTHONPATH="$PYTHONPATH:/app"

EXPOSE 8888

