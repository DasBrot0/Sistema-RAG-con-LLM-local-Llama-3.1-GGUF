# ====================================================================
# IMAGEN CUDA 12.6.0
# ====================================================================

# Usamos la 12.6.0 'devel' (desarrollo) porque necesitamos compilar
# y también correr las librerías de cuBLAS.
FROM nvidia/cuda:12.6.0-cudnn-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
WORKDIR /app
ENV VIRTUAL_ENV=/app/venv
# Asegura que el PATH incluya el binario del venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH" 

# Instala dependencias del sistema esenciales
RUN apt-get update && apt-get install -y \
    python3.11 \
    python3.11-venv \
    python3.11-dev \
    python3-pip \
    git \
    build-essential \
    libpq-dev \
    postgresql-client \
    && rm -rf /var/lib/apt/lists/*

# Configura Python
RUN update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1 && \
    update-alternatives --install /usr/bin/pip pip /usr/bin/pip3 1

# Crea y configura venv
RUN python -m venv $VIRTUAL_ENV
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

# 1. Instala PyTorch con CUDA
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu126

# 2. Instala requirements.txt
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# 3. Instalación de llama-cpp-python
RUN export CMAKE_ARGS="-DGGML_CUDA=on -DCUDA_PATH=/usr/local/cuda-12.6 -DCUDAToolkit_ROOT=/usr/local/cuda-12.6 -DCUDAToolkit_INCLUDE_DIR=/usr/local/cuda-12.6/include -DCUDAToolkit_LIBRARY_DIR=/usr/local/cuda-12.6/lib64" && \
    export CUDACXX=/usr/local/cuda-12.6/bin/nvcc && \
    pip install llama-cpp-python --no-cache-dir --force-reinstall --upgrade

# 4. Copia el resto de la aplicación
COPY initial_rag.py .
COPY backend_rag.py .
COPY .env .
COPY models/ /app/models/
COPY data/ /app/data/

EXPOSE 8000