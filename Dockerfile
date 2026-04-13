FROM python:3.11-slim

# System deps: curl (used in start.sh), build tools (faiss-cpu, HuggingFace tokenizers)
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps first (layer cache friendly — only rebuilds on requirements change)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY src/ src/
COPY api/ api/
COPY ui/ ui/
COPY evaluation/ evaluation/
COPY main.py .

# Pre-create volume-mount targets so container owns them from the start
RUN mkdir -p faiss_index evaluation/results

COPY start.sh .
RUN chmod +x start.sh

EXPOSE 8000 8501

CMD ["./start.sh"]
