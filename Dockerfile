# Python 3.10 works with all recent TensorFlow 2.x releases
FROM python:3.10-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /app

# System dependencies for pillow/tensorflow
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

# Default: full pipeline in one container (docker-compose overrides per stage)
CMD ["sh", "-c", "python download_data.py && python train_dog_cat.py && python benchmark_hf.py"]
