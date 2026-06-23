FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

WORKDIR /runner
ENV RUNNER_MODE=edit \
    MODEL_ID=Qwen/Qwen-Image-Edit \
    CITIVIA_DATA_DIR=/workspace \
    PORT=8000 \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/workspace/.cache/huggingface

COPY runners/common/requirements.txt ./requirements.txt
RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates git \
    && pip install --no-cache-dir -r requirements.txt \
    && apt-get purge -y --auto-remove git \
    && rm -rf /var/lib/apt/lists/*

COPY runners/common/runner.py ./runner.py
VOLUME ["/workspace"]
EXPOSE 8000
CMD ["python", "runner.py"]
