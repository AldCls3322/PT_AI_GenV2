# AI RAG Assitant Banking Clients

# Objetivo

Este proyecto tiene el objetivo de crear un MVP de un asistente inteligente para clientes de un banco.

Utiliza Python, LangChain, RAG con ChromaDB y variables parametizables para el uso de LLMs de OpenAI o Ollama. 

Es funcional con API REST, no tiene interfaz grafica, fue implementado con FastAPI.

---

## Descripcion

Principales temas de architectura

- **Orquestador** con LangChain.
- **RAG (Retrieval-Augmented Generation)** para fundamentar respuestas, acepta documentos txt y pdf.
- **Memoria de conversación** memoria por sesión.
- **Clasificación automática** de intención hacia 5 procesos bancarios A-E.
    A. Clarification handling / Inquiry resolution
    B. Product cancellation
    C. Incident escalation
    D. Customer data update
    E. Internal complaint management

- **LLM** (OpenAI o Ollama), se tiene de default GPT-4o.

---

## Requisitos

- **Python 3.12**
- **API Key de OpenAI** habilitado `gpt-4o` o **Ollama** installarlo localmente.
- **Windows**

---

## Instalación

### 1. Clonar proyecto

```
git clone <repo-url>
```

### Crea venv

```powershell
python -m venv rag_pt_ai
.\rag_pt_ai\Scripts\Activate.ps1
```

### Instala dependencias

```powershell
pip install -r requirements.txt
```

```powershell
pip install -e .
```

## Configuraciones

Ver archivo `.env`. Aqui ingresas tu API_KEY y el modelo que quieres usar. Igual en `./app/config.py`

1. `CHUNK_SIZE` = `500`. Tamaño en caracteres de cada fragmento (`200` – `1200`)
2. `CHUNK_OVERLAP` = `50`. Overlap entre chunks consecutivos (`0` – `200`)
3. `EMBEDDING_MODEL` = `all-MiniLM-L6-v2`. Modelo de embeddings local. 
4. `EMBEDDED_SIZING` = `384`. Alineado con el Embedded model, su dimensionamiento es de `384` por Chunk.
5. `TOP_K` = `4`. Número de chunks recuperados por consulta | `2` – `10` |
6. `OPENAI_API_KEY` = `string`. API Key para que se puedan procesar los requests.
7. `LLM_MODEL` = `"gpt-4o"`. Modelo LLM para crear la solucion de los requests.
8. `Vectore store` = `Chroma` . Se uso Chroma.

## Correr proyecto

```powershell
Remove-Item -Recurse -Force chroma_store
python run.py

# Tambien podrias
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

---

## API

### `POST /api/v1/chat`

Principal prueba. Envia un mensaje del cliente y recibe la respuesta del agente.

**Request Body:**

```json
{
    "conversation_id": "conv-001",
    "user_id": "usr-42",
    "message": {
        "text": "¿Cómo cancelo mi tarjeta de crédito?"
    },
    "metadata": {
        "channel": "app",
        "timestamp": "2025-01-15T10:00:00Z"
    }
}
```

**Response `200 OK`:**

```json
{
    "conversation_id": "conv-001",
    "user_id": "usr-42",
    "reply": "Para cancelar tu tarjeta de crédito Banorte, primero debes liquidar el saldo...",
    "detected_process": "B — Product Cancellation",
    "process_info": {
        "process_id": 2,
        "process_name": "Product Cancellation",
        "team_area_responsible": "Retention & Cancellations Team",
        "average_time_till_solved": "24h",
        "atention_channel": "app, web, branch",
        "priority_level": "high"
    },
    "sources": ["process_b_cancellation.txt"],
    "timestamp": "2025-01-15T10:00:01.234Z"
}
```

---

### `POST /api/v1/ingest`

Dispara manualmente el pipeline RAG sobre la carpeta `docs/`. No requiere body.

```bash
POST http://localhost:8000/api/v1/ingest
```

**Response:**
```json
{
    "status": "success",
    "processed": 2,
    "details": [
        {"file": "process_b_cancellation.txt", "process_code": "B", "chunk_count": 11},
        {"file": "process_d_customer_data_update.txt", "process_code": "D", "chunk_count": 24}
    ]
}
```

---

## RAG

El sistema RAG (Retrieval-Augmented Generation) enriquece las respuestas.

### Estrategia de Búsqueda

Se uso **Cosine Similarity** como estrategia de búsqueda vectorial, implementada en `app/rag/retriever.py`.

### Agregar Documentos al RAG

1. Coloca archivos `.txt` o `.pdf` en la carpeta `docs/`
2. Puedes podificar chunking y overlap por si documentos son mas largos.
3. Reinicia el servidor o llama a `POST /api/v1/ingest`
4. Los documentos ya procesados se omiten automáticamente (deduplicación por nombre de archivo)

## 🗄️ Base de Datos

El proyecto usa **SQLite** (archivo `banorte.db`) para desarrollo. Si se quiere evolucionar de MVP se puede migrar a PostgreSQL cambiando `DATABASE_URL` en `.env`.


---

## Memoria de Conversación

Mantiene memoria por sesion. Implementado en `app/memory/chat_memory.py` usando LangChain `InMemoryChatMessageHistory`.
