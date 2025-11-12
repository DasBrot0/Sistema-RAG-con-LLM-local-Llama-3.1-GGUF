import os
import time
from dotenv import load_dotenv

# Librerías LLM/LangChain para GGUF
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_postgres.vectorstores import PGVector
from langchain_core.output_parsers import StrOutputParser
from langchain_community.chat_models import ChatLlamaCpp
from langchain_core.prompts import ChatPromptTemplate
from transformers import pipeline
from typing import List
from sqlalchemy.ext.asyncio import create_async_engine

# Librerías de FastAPI
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from contextlib import asynccontextmanager
import uvicorn

# ==============================
# Configuración inicial
# ==============================

load_dotenv()

COLLECTION_NAME = os.getenv("COLLECTION_NAME")

CONNECTION_STRING = os.getenv("DATABASE_URL_ASYNC") 

MODEL_PATH = os.getenv("MODEL_PATH")

LANG = os.getenv("LANG")

EMBEDDING_MODEL = "nomic-ai/nomic-embed-text-v1.5"

# Variables globales para la aplicación
llm = None
rag_chain = None
translator_pipeline = None
embeddings = None
db_engine = None

# ============================================================
# Clase personalizada para embeddings
# ============================================================

class ModEmbeddings(HuggingFaceEmbeddings):
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Añade el prefijo 'search_document:' a los documentos."""
        prefixed_texts = [f"search_document: {text}" for text in texts]
        return super().embed_documents(prefixed_texts)
    
    def embed_query(self, text: str) -> List[float]:
        """Añade el prefijo 'search_query:' a las preguntas (query)."""
        prefixed_text = f"search_query: {text}"
        return super().embed_query(prefixed_text)

# ==============================
# Manejo de español/inglés
# ==============================

def translate_query(question: str) -> str:
    global translator_pipeline

    if LANG == "es" and translator_pipeline:
        try:
            result = translator_pipeline(question)
            translated_text = result[0]['translation_text']
            print(f"[INFO] Pregunta traducida (es → en): '{translated_text}'")
            return translated_text
        except Exception as e:
            print(f"[ERROR] Error durante la traducción: {e}")
            # Si falla la traducción, seguir con la pregunta original
            return question
    
    print(f"[INFO] Pregunta no traducida (idioma detectado: {LANG})")
    return question

def get_system_prompt() -> str:
    if LANG == "es":
        return """
        Eres un asistente experto en la plataforma y metodología de **Gamit!**.
        Tu misión principal es **ayudar al usuario a comprender** la plataforma Gamit!, su diseño, implementación y resultados, basándote *únicamente* en la información contenida en el contexto o los documentos proporcionados.
        
        INSTRUCCIONES Y REGLAS ESTRICTAS:
        1.  **Fundamento de Contexto:** Tu respuesta debe construirse *exclusivamente* con la información textual presente en el contexto(s) y archivo(s) proporcionado(s).
        2.  **Análisis Exhaustivo:** Analiza y procesa *todo* el contexto relevante antes de formular la respuesta.
        3.  **Síntesis Coherente:** Si la información está distribuida o fragmentada en múltiples secciones del contexto, sintetízala de manera coherente y fluida para asegurar la comprensión del tema.
        4.  **Respuesta Completa:** Para preguntas amplias o de concepto sobre Gamit!, proporciona una respuesta completa y profunda, utilizando toda la información relevante disponible.
        5.  **Reorganización/Reformulación:** Puedes reorganizar y reformular la información para mayor claridad y un tono experto, pero **está prohibido inventar, inferir o añadir datos** que no estén literalmente en el contexto.
        6.  **Insuficiencia de Información:** Si la información necesaria para responder la pregunta no está presente en el contexto, o es insuficiente, debes indicarlo claramente, especificando qué datos o detalles faltan.
        7.  **Formato y Estructura:**
            * Responde en **español** de forma clara, concisa pero completa, y bien estructurada.
            * Utiliza **Markdown (negritas, listas, encabezados)** para enriquecer la presentación.

        8.  **Prohibiciones:**
            * No menciones el número, nombre o fuente del fragmento/archivo del que extrajiste la respuesta.
            * Nunca cites la fuente ni uses referencias.
        """
    elif LANG == "en":
        return """
        You are an expert assistant in the Gamit! platform and methodology.
        Your main mission is to help the user understand the Gamit! platform, its design, implementation, and results, based solely on the information contained in the provided context or documents.

        STRICT INSTRUCTIONS AND RULES:
        1. Context-Based Foundation: Your response must be built exclusively on the textual information present in the provided context(s) and file(s).
        2. Exhaustive Analysis: Analyze and process all relevant context before formulating your response.
        3. Coherent Synthesis: If the information is distributed or fragmented across multiple sections of the context, synthesize it coherently and fluently to ensure understanding of the topic.
        4. Complete Response: For broad or conceptual questions about Gamit!, provide a complete and thorough answer using all relevant information available.
        5. Reorganization/Reformulation: You may reorganize and reformulate the information for clarity and expert tone, but it is forbidden to invent, infer, or add data not literally present in the context.
        6. Insufficient Information: If the necessary information to answer the question is not present in the context, or is insufficient, you must state this clearly, specifying which data or details are missing.
        7. Format and Structure:
        * Respond in English clearly, concisely but completely, and in a well-structured manner.
        * Use Markdown (bold, lists, headings) to enhance presentation.
        8. Prohibitions:
        * Do not mention the number, name, or source of the fragment/file from which you extracted the answer.
        * Never cite the source or use references.
        """
    else:
        # Default a Español si falla la detección
        return "Eres un asistente experto. Responde en español clara y profesionalmente."

# ==============================
# Lógica de RAG (Funciones de soporte)
# ==============================

def format_docs(docs):
    """Formatea documentos eliminando duplicados y añadiendo separadores claros"""
    unique_contents = []
    seen = set()
    
    for i, doc in enumerate(docs):
        content = doc.page_content.strip()
        if content and content not in seen:
            seen.add(content)
            unique_contents.append(f"[Fragmento {i+1}]\n{content}")
    
    return "\n\n---\n\n".join(unique_contents)
    
def clean_response(text: str) -> str:
    # Busca el último punto completo antes de un posible corte
    last_period = max(text.rfind("."), text.rfind("!"), text.rfind("?"))
    if last_period != -1 and last_period > len(text) * 0.6: 
        return text[:last_period+1]
    return text

def debug_chain_input(input_dict: dict) -> dict:
    """Imprime el estado de las claves 'context' y 'question' antes de construir el prompt."""
    
    context = input_dict.get("context", "Contexto no encontrado.")
    question = input_dict.get("question", "Pregunta original no encontrada.")
    
    context_preview = context.split('\n\n---\n\n')[0][:300] + "..." if context else "VACÍO"
    
    print("\n--- INICIO DEBUG DE LA CADENA RAG ---")
    print(f"✔️ PREGUNTA ORIGINAL para el LLM: {question}")
    print(f"✔️ CONTEXTO RECUPERADO:")
    print(context_preview)
    print("---------------------------------------")
    
    return input_dict

# ==============================
# Cargar modelo principal (LlamaCpp - GGUF)
# ==============================

def create_gguf_llm(model_path):
    print(f"[INFO] Cargando modelo GGUF (ChatModel) local desde: {model_path}")
    
    N_GPU_LAYERS = -1
    N_THREADS = 8

    try:
        llm_instance = ChatLlamaCpp(
            model_path=model_path,
            n_gpu_layers=N_GPU_LAYERS, 
            n_ctx=8192,
            temperature=0.1,
            verbose=True,
            max_tokens=2048,
            n_batch=2048,
            n_threads=N_THREADS,
            repeat_penalty=1.15,
            stop=["<|eot_id|>", "</s>"],
            echo=False,
            streaming=False,
            chat_format="llama3"
        )
        print("[INFO] Modelo GGUF (ChatModel) cargado correctamente con LlamaCpp.")
        return llm_instance
    except Exception as e:
        print("######################################################")
        print(f"[ERROR FATAL] Error al cargar LlamaCpp (ChatModel). Detalle: {e}")
        print("######################################################")
        raise e

# ==============================
# Inicialización de la aplicación
# ==============================

def initialize_rag_system():
    global llm, rag_chain, translator_pipeline, embeddings, db_engine
    
    # 0. Cargar Embeddings (Nomic)
    print("[INFO] Cargando modelo de Embeddings (Nomic)...")
    embeddings = ModEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'trust_remote_code': True}
    )
    
    # 1. Conexión a la base de datos vectorial
    print(f"[INFO] Conectando a la base de datos vectorial: {COLLECTION_NAME}")
    try:
        db_engine = create_async_engine(CONNECTION_STRING)
    except Exception as e:
        print(f"[ERROR FATAL] No se pudo crear el motor asíncrono: {e}")
        raise e
    
    vectorstore = PGVector(
        embeddings=embeddings,
        collection_name=COLLECTION_NAME,
        connection=db_engine,
        create_extension=False
    )

    # 2. Configuración del retriever
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={ "k": 8, "fetch_k": 25, "lambda_mult": 0.6 }
    )
    print("[INFO] BD vectorial lista y retriever configurado.")
    
    # 3. LIBERAR VRAM DE EMBEDDINGS¿
    print("[INFO] Liberando memoria del modelo de embeddings.")
    del embeddings
    print("[INFO] Memoria GPU de embeddings liberada.")
        
    # 4. Inicializar traductor
    print("[INFO] Cargando modelo de traducción (Helsinki-NLP)...")
    try:
        translator_pipeline = pipeline("translation", model="Helsinki-NLP/opus-mt-es-en", device=0)
        print("[INFO] Modelo de traducción cargado en GPU (device=0).")
    except Exception as e:
        translator_pipeline = None
        print("######################################################")
        print(f"[ADVERTENCIA] No se pudo cargar el traductor en GPU. Error: {e}")
        print("[ADVERTENCIA] La traducción ES-EN estará desactivada.")
        print("######################################################")

    # 5. Cargar modelo principal
    llm = create_gguf_llm(MODEL_PATH)

    # 6. CREACIÓN DEL NUEVO PROMPT TEMPLATE
    
    # Obtiene el prompt del sistema
    system_prompt_content = get_system_prompt().strip()
    
    # Define la plantilla humana
    human_template_content = """Contexto:
{context}

Pregunta: {question}"""

    # Construye el ChatPromptTemplate
    prompt_template = ChatPromptTemplate.from_messages([
        ("system", system_prompt_content),
        ("human", human_template_content)
    ])

    print("[INFO] ChatPromptTemplate creado.")

    # =================================================================
    # 7. NUEVA CADENA RAG (MÁS LIMPIA)
    # =================================================================
    rag_chain = (
        {
            "context": RunnablePassthrough() | RunnableLambda(translate_query) | retriever | format_docs,
            "question": RunnablePassthrough()
        }
        | RunnableLambda(debug_chain_input) # Debug
        | prompt_template
        | llm 
        | StrOutputParser()
        | RunnableLambda(clean_response)
    )

    print("[INFO] Cadena RAG (con ChatPromptTemplate) creada y sistema inicializado.")
    
# ==============================
# Definición del Backend FastAPI
# ==============================

# Definir el modelo de Pydantic para la entrada
class ChatRequest(BaseModel):
    question: str
    
# Definir el modelo de Pydantic para la salida
class ChatResponse(BaseModel):
    answer: str
    time_taken_s: float
    
# Evento de inicio: Cargar todos los modelos
@asynccontextmanager
async def lifespan(app: FastAPI):
    global db_engine
    """
    Maneja los eventos de inicio y apagado de la aplicación (Lifespan).
    """
    print("=========================================================")
    print("Iniciando el sistema RAG...")
    # --- CÓDIGO DE INICIO (Startup) ---
    initialize_rag_system() 
    print("Sistema RAG listo para recibir peticiones.")
    print("=========================================================\n")
    
    yield # Aquí la aplicación empieza a recibir peticiones
    
    # --- CÓDIGO DE APAGADO (Shutdown) ---
    print("\n[INFO] El servidor se está apagando. Liberando recursos...")
    if db_engine:
        await db_engine.dispose() # Cierre limpio de la conexión asíncrona
        print("[INFO] Motor de DB asíncrono cerrado.")

# Inicialización de la app usando el lifespan definido
app = FastAPI(
    title="RAG Chatbot API",
    description="Backend para el chatbot RAG sobre la plataforma Gamit!",
    lifespan=lifespan
)

# Endpoint para interactuar con el RAG
@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    global rag_chain
    
    if rag_chain is None:
        raise HTTPException(status_code=503, detail="El sistema RAG aún no se ha inicializado o falló al cargar.")

    try:
        start_time = time.time()
        response = await rag_chain.ainvoke(request.question)
        
        end_time = time.time()
        
        return ChatResponse(
            answer=response,
            time_taken_s=end_time - start_time
        )
    except Exception as e:
        print(f"Error durante la ejecución de RAG: {e}")
        # Notificar al cliente con un 500
        raise HTTPException(status_code=500, detail=f"Error interno del servidor: {e}")

# ==============================
# Ejecución del servidor
# ==============================

if __name__ == "__main__":
    
    print("\n[INFO] Iniciando servidor Uvicorn...")
    uvicorn.run("backend_rag:app", host="0.0.0.0", port=8000, reload=False)