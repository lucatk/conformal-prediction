FROM runpod/pytorch:0.7.2-dev-ubuntu2404-cu1281-torch271

WORKDIR /workspace

RUN apt-get update && apt-get install --no-install-recommends -y \
        build-essential && \
    	apt clean && rm -rf /var/lib/apt/lists/*

ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Activate virtual environment from base image
COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY . .

CMD ["streamlit", "run", "/workspace/app.py"]
