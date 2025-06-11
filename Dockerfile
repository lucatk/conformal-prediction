FROM nvidia/cuda:12.9.0-cudnn-runtime-ubuntu24.04 AS builder

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1
WORKDIR /app

RUN apt-get update && \
    apt-get install -y python3-pip python3-dev python3-venv python-is-python3 && \
    rm -rf /var/lib/apt/lists/*

#FROM nvidia/cuda:12.9.0-cudnn-runtime-ubuntu24.04
#WORKDIR /app

#ENV ADIENCE_USER=adiencedb
#ENV ADIENCE_PASS=adience
#ENV ADIENCE_URL='http://www.cslab.openu.ac.il/download/adiencedb/AdienceBenchmarkOfUnfilteredFacesForGenderAndAgeClassification'
#
#RUN apt update && apt install -y wget
#
#RUN mkdir -p /app/.datasets && cd /app/.datasets && \
#    wget --user $ADIENCE_USER --password $ADIENCE_PASS $ADIENCE_URL/fold_0_data.txt && \
#    wget --user $ADIENCE_USER --password $ADIENCE_PASS $ADIENCE_URL/fold_1_data.txt && \
#    wget --user $ADIENCE_USER --password $ADIENCE_PASS $ADIENCE_URL/fold_2_data.txt && \
#    wget --user $ADIENCE_USER --password $ADIENCE_PASS $ADIENCE_URL/fold_3_data.txt && \
#    wget --user $ADIENCE_USER --password $ADIENCE_PASS $ADIENCE_URL/fold_4_data.txt && \
#    wget --user $ADIENCE_USER --password $ADIENCE_PASS $ADIENCE_URL/aligned.tar.gz

RUN python -m venv .venv
COPY requirements.txt ./
RUN .venv/bin/pip install -r requirements.txt

#COPY --from=builder /app/.venv .venv/
COPY . .
CMD ["/app/.venv/bin/streamlit", "run", "app.py"]
