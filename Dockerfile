FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime AS builder

WORKDIR /workspace

RUN python -m venv .venv
COPY requirements.txt ./
RUN .venv/bin/pip install -r requirements.txt

FROM python:3.11

COPY --from=builder /workspace/.venv .venv/
COPY . .
CMD ["/workspace/.venv/bin/streamlit", "run", "app.py"]
