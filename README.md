# RAG-GGUF (Llama 3.1 GGUF + PGVector)

## Preparación previa

### Prerequisitos

* Internet
* **GPU NVIDIA** (8GB+ VRAM).
* **RAM** (32GB+ RAM).
* **Drivers NVIDIA** actualizados (se probó con la versión 576.97).
* **Docker** y **Docker Compose** instalados.
* **Nvidia-container-toolkit** (para que Docker pueda ver la GPU).

### Instalación

**Descarga de modelo**

https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/blob/main/Meta-Llama-3.1-8B-Instruct-Q6_K.gguf

* Después de la descarga, copiar el modelo y pegarlo en la carpeta /models

**Construir la Imagen de Docker:**

* **¡Ten Paciencia!** Tarda entre **15 y 25 minutos** la primera vez. Es normal.
    
```bash
docker-compose build
```

### Despliegue

**Levantar los Servicios:**
    
* Este comando iniciará la base de datos PGVector y la API de RAG.

```bash
docker-compose up -d
```

## Uso de la API

Una vez que los contenedores estén corriendo, el sistema hará dos cosas:
1.  **Ingesta:** El script `initial_rag.py` se ejecutará la primera vez, procesará el PDF de `data/` y lo guardará en PGVector.
2.  **Servidor:** El servidor FastAPI (`backend_rag.py`) se iniciará en el puerto 8000.

### Endpoint: `POST /chat`

* Abrir http://127.0.0.1:8000/docs en cualquier navegador (Swagger UI probado en Google Chrome)
* Click en el botón "Try it out"
* Cambiar "string" por la pregunta
* Bajar hasta Details - Response body y saldrá la respuesta en el siguiente formato:

```json
{
  "answer": "Gamit! es una plataforma experta en... ",
  "time_taken_s": 8.45
}
```

### Estructura del proyecto:

```text
.
├── data/                   # Documento para la ingesta
├── models/                 # Modelo .gguf
├── .env                    # Variables de entorno
├── .gitignore              # Archivos a ignorar en la subida al repo
├── backend_rag.py          # Script de FastAPI (la API principal)
├── docker-compose.yml      # Orquestador para levantar la DB y la App
├── Dockerfile              # Archivo para construir el contenedor Docker
├── init_pgvector.sql       # Script SQL para inicializar la extensión PGVector
├── initial_rag.py          # Script inicial de ingesta (carga documentos a PGVector)
├── README.md               # Este archivo
└── requirements.txt        # Dependencias de Python (sin llama-cpp-python ni torch)
```

### Nota:

* Las preguntas en español se traducen al inglés para ayudar al retriever, pero la traducción no es exacta.
* Al mencionar a Gamit, se debe escribir tal cual con la primera letra en mayúscula ("Gamit", no "gamit").
