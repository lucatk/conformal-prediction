FROM nvidia/cuda:12.8.0-base-ubuntu24.04

WORKDIR /workspace

RUN apt-get update && apt-get install --no-install-recommends -y \
        build-essential \
        python3.10 \
        python3-pip && \
    	apt clean && rm -rf /var/lib/apt/lists/*

RUN pip3 --no-cache-dir install torch torchvision torchaudio \
        --index-url https://download.pytorch.org/whl/cu128

COPY requirements.txt ./
RUN pip3 install -r requirements.txt

COPY . .

CMD ["streamlit", "run", "app.py"]
