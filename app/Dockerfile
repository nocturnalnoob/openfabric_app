FROM openfabric/tee-python-cpu:dev

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Create directories for models and data
RUN mkdir -p /app/models /app/datastore

# Copy only necessary files for Poetry installation
COPY pyproject.toml ./

# Install dependencies using Poetry
RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --upgrade poetry && \
    python3 -m poetry install --only main && \
    rm -rf ~/.cache/pypoetry/{cache,artifacts}

# Download LLM model (modify URL based on your needs)
RUN mkdir -p models && \
    wget -O models/llama-2-7b.gguf https://huggingface.co/TheBloke/Llama-2-7B-GGUF/resolve/main/llama-2-7b.Q4_K_M.gguf

# Copy the rest of the source code into the container
COPY . .

# Expose port 5000 for the Flask app
EXPOSE 8888

# Start the Flask app using the start.sh script
CMD ["sh","start.sh"]