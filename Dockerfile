FROM python:3.11-slim

WORKDIR /app

# curl: needed to fetch the model from GitHub Releases at build time.
# The model file exceeds GitHub's web upload limit, so it is distributed
# as a release asset rather than committed to the repository.
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .

# Download the ONNX model. Replace the URL with your own release asset URL.
RUN curl -fL -o talmo_norwood_v1.onnx \
    "https://github.com/KKKKKhsbiran/Talmo-Project/releases/download/v1.0.0/talmo_norwood_v1.onnx" \
    && ls -lh talmo_norwood_v1.onnx

CMD uvicorn main:app --host 0.0.0.0 --port ${PORT:-8000}
