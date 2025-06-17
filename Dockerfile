FROM nvidia/cuda:12.8.0-base-ubuntu24.04

WORKDIR /workspace

RUN apt-get update && apt-get install --no-install-recommends -y \
        build-essential \
        python3.12 \
        python3-pip \
        python3.12-venv \
        python-is-python3 && \
    	apt clean && rm -rf /var/lib/apt/lists/*

RUN python -m venv .venv

RUN .venv/bin/pip --no-cache-dir install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu128

COPY requirements.txt ./
RUN .venv/bin/pip install -r requirements.txt

COPY . .

CMD ["/workspace/.venv/bin/streamlit", "run", "/workspace/app.py"]
