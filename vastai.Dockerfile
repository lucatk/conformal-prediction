FROM vastai/pytorch:2.7.0-cuda-12.8.1-py312-24.04

WORKDIR /opt/workspace-internal/

# Activate virtual environment from base image
RUN . /venv/main/bin/activate

COPY requirements.txt ./
RUN pip install -r requirements.txt

COPY . .

#CMD ["streamlit", "run", "/opt/workspace-internal/app.py"]
