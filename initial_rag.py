import os
import time
import torch
from dotenv import load_dotenv
from typing import List

# Librerías necesarias para Ingestión
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres.vectorstores import PGVector

#==============================
# Configuración inicial
#==============================

load_dotenv()

# Variables requeridas para la DB y el PDF
COLLECTION_NAME = os.getenv("COLLECTION_NAME")
CONNECTION_STRING = os.getenv("DATABASE_URL_SYNC") 
PDF_PATH = os.getenv("PDF_PATH")

EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5"

#============================================================
# Clase personalizada para embeddings
#============================================================

class ModEmbeddings(HuggingFaceEmbeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Añade el prefijo 'search_document:' a los documentos."""
        prefixed_texts = [f"search_document: {text}" for text in texts]
        return super().embed_documents(prefixed_texts)
    
    def embed_query(self, text: str) -> List[float]:
        """Añade el prefijo 'search_query:' a las preguntas (query)."""
        prefixed_text = f"search_query: {text}"
        return super().embed_query(prefixed_text)

#==============================
# Flujo de Ingestión Exclusiva
#==============================

if __name__ == "__main__":
    
    start_time_total = time.time()
    
    print("=========================================================")
    print("🚀 Iniciando Proceso de Ingestión de Embeddings (PGVector)")
    print("=========================================================")
    print(f"[INFO] Documento a procesar: {PDF_PATH}")
    
    # 1. Carga del Documento
    try:
        loader = PyPDFLoader(PDF_PATH)
        docs = loader.load()
    except Exception as e:
        # Mantengo la estructura de error para guiar al usuario
        print(f"######################################################")
        print(f"[ERROR FATAL] Fallo al cargar el PDF: {e}")
        print(f"Asegúrate de que la ruta '{PDF_PATH}' sea correcta y el archivo exista.")
        print("######################################################")
        exit(1)
        
    print(f"[INFO] Documento cargado exitosamente. Total de páginas: {len(docs)}.")

    # 2. Inicializar Embeddings
    embeddings = ModEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'trust_remote_code': True}
    )
    print(f"[INFO] Modelo de Embeddings cargado: {EMBEDDING_MODEL}")

    # 3. Estrategia de chunking
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000, 
        chunk_overlap=300,
        separators=[
            "\n\n\n", 
            "\n\n", 
            "\n", 
            ". ", 
            ", ", 
            " ",
            ""
        ],
        length_function=len,
    )
    splits = splitter.split_documents(docs)
    print(f"[INFO] Documento dividido en {len(splits)} fragmentos (chunks).")

    # 4. Enriquecimiento de metadatos (Opcional, pero recomendado)
    for i, doc in enumerate(splits):
        doc.metadata["chunk_index"] = i
        doc.metadata["char_count"] = len(doc.page_content)
        # Extrae primeras palabras como resumen (útil para debug)
        words = doc.page_content.split()
        doc.metadata["summary"] = ' '.join(words[:20]) if len(words) > 20 else doc.page_content

    doc_ids = [str(i) for i in range(len(splits))]

    # 5. Guardar en PGVector (Crea los embeddings y los almacena)
    print(f"[INFO] Conectando a PostgreSQL y guardando {len(splits)} documentos en '{COLLECTION_NAME}'.")
    
    ingestion_start_time = time.time()
    
    vectorstore = PGVector.from_documents(
        documents=splits,
        embedding=embeddings,
        collection_name=COLLECTION_NAME,
        connection=CONNECTION_STRING, 
        ids=doc_ids,
        pre_delete_collection=True 
    )

    ingestion_end_time = time.time()

    # 6. Limpieza de memoria
    if torch.cuda.is_available():
        print("[INFO] Embeddings calculados y almacenados. Liberando memoria del modelo de embeddings.")
        del embeddings
        torch.cuda.empty_cache()
        
    end_time_total = time.time()

    print("\n=========================================================")
    print(f"✨ Proceso de Ingestión de Embeddings Finalizado")
    print(f"   -> Fragmentos almacenados: {len(splits)}")
    print(f"   -> Tiempo de Ingestión (DB): {ingestion_end_time - ingestion_start_time:.2f}s")
    print(f"   -> Tiempo Total del Script: {end_time_total - start_time_total:.2f}s")
    print("=========================================================")