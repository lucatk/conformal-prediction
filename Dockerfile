FROM nvidia/cuda:12.8.0-base-ubuntu24.04

WORKDIR /workspace

RUN apt-get update && apt-get install --no-install-recommends -y \
        build-essential \
        python3.12 \
        python3-pip \
        python-is-python3 && \
    	apt clean && rm -rf /var/lib/apt/lists/*

RUN pip --no-cache-dir install --break-system-packages torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu128

COPY requirements.txt ./
RUN pip install --break-system-packages -r requirements.txt

COPY . .

CMD ["streamlit", "run", "/workspace/app.py"]
