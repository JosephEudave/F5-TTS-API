FROM nvcr.io/nvidia/tritonserver:24.02-py3

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    python3-pip \
    python3-venv \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /workspace

# Create and activate Python virtual environment
RUN python3 -m venv /opt/venv
ENV PATH="/opt/venv/bin:$PATH"

# Clone F5-TTS repository
RUN git clone https://github.com/Frydesk/F5-TTS.git \
    && cd F5-TTS \
    && git submodule update --init --recursive

# Install specific versions of core dependencies
RUN pip install --no-cache-dir \
    torch==2.7.0 \
    torchaudio==2.7.0 \
    fastapi==0.115.12 \
    uvicorn==0.34.2 \
    python-multipart==0.0.20 \
    safetensors==0.5.3 \
    pyyaml==6.0.2 \
    numpy==2.2.5 \
    starlette==0.46.2 \
    click==8.2.0 \
    pydantic==2.11.4

# Install F5-TTS in the virtual environment
RUN cd F5-TTS \
    && pip install -e .

# Set environment variables
ENV PYTHONPATH=/workspace/F5-TTS:$PYTHONPATH
ENV TRITON_PYTHON_EXECUTABLE=/opt/venv/bin/python

# Copy model repository
COPY model_repository /models

# Expose Triton ports
EXPOSE 8000 8001 8002

# Start Triton server
CMD ["tritonserver", "--model-repository=/models"]
