FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime AS builder

WORKDIR /tmp/build

RUN #python -m venv .venv
COPY requirements.txt ./
RUN #.venv/bin/pip install -r requirements.txt

RUN pip install --no-cache-dir --upgrade pip \
 && pip wheel --wheel-dir=/tmp/wheels -r requirements.txt

FROM pytorch/pytorch:2.7.1-cuda12.8-cudnn9-runtime

WORKDIR /workspace

COPY --from=builder /tmp/wheels /tmp/wheels
COPY requirements.txt ./

# Install from local wheels
RUN pip install --no-cache-dir --no-index --find-links=/tmp/wheels -r requirements.txt \
 && rm -rf /tmp/wheels

# Copy only what you need
COPY . .

CMD ["streamlit", "run", "app.py"]
