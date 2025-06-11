FROM python:3.12.9 AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
WORKDIR /app


RUN python -m venv .venv
COPY requirements.txt ./
RUN .venv/bin/pip install -r requirements.txt
FROM python:3.12.9-slim
WORKDIR /app

ENV ADIENCE_USER=adiencedb \
    ADIENCE_PASS=adience \
    ADIENCE_URL='http://www.cslab.openu.ac.il/download/adiencedb/AdienceBenchmarkOfUnfilteredFacesForGenderAndAgeClassification'

RUN apt install -y cuda-nvcc-12-2 libcublas-12-2 libcudnn8 wget
RUN wget --user $ADIENCE_USER --password $ADIENCE_PASS $ADIENCE_URL/fold_0_data.txt && \
    wget --user $ADIENCE_USER --password $ADIENCE_PASS $ADIENCE_URL/fold_1_data.txt && \
    wget --user $ADIENCE_USER --password $ADIENCE_PASS $ADIENCE_URL/fold_2_data.txt && \
    wget --user $ADIENCE_USER --password $ADIENCE_PASS $ADIENCE_URL/fold_3_data.txt && \
    wget --user $ADIENCE_USER --password $ADIENCE_PASS $ADIENCE_URL/fold_4_data.txt && \
    wget --user $ADIENCE_USER --password $ADIENCE_PASS $ADIENCE_URL/aligned.tar.gz

COPY --from=builder /app/.venv .venv/
COPY . .
CMD ["/app/.venv/bin/streamlit", "run", "app.py"]
