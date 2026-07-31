# Base image: lightweight official Python
FROM python:3.12-slim

# Set working directory inside the container
WORKDIR /app

# Copy only requirements first (so Docker can cache this layer
# and skip reinstalling packages if only your code changes later)
COPY requirements.txt .

# Install CPU-only torch first (much smaller/faster than the default CUDA build)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

# Download the nltk tokenizer data needed by preprocess.py
RUN python -m nltk.downloader punkt_tab

# Now copy the rest of the project files
COPY api.py model.py preprocess.py model.pth vocab.json ./

# Render/Railway inject the port to use via $PORT; default to 8000 for local runs
ENV PORT=8000
EXPOSE 8000

# Start the API. Using sh -c so $PORT is resolved from the environment
CMD ["sh", "-c", "uvicorn api:app --host 0.0.0.0 --port ${PORT}"]