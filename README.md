# Literature Assistant — README

## Project purpose
1. Provide a publication-based RAG (retrieval-augmented generation) pipeline for querying a Zotero publication collection in natural language.
2. Offer a fallback external paper search via the Semantic Scholar API to expand context when the local vector DB lacks sufficient information.
3. Support evaluation of retrieval and answer quality across curated queries.

## Key features
1. Grounded LLM answers with source citations (paper title, section).
2. PDF processing, section-aware chunking and vector embedding ingestion.
3. Vector DB storage using Chroma and configurable embedder backends (e.g., BERT).
4. External paper recommendation via Semantic Scholar with a multistage search strategy.
5. Evaluation tooling for chunk-level, multi-paper and answer-quality metrics.

## Requirements
1. Python 3.11.
2. `pip` for dependency installation.
3. Local Chroma DB path (default `./backend/chroma_db`).
4. Ollama running locally for the LLM endpoint when using `mistral-nemo` (or configure another supported LLM).
5. Optional API keys for `Zotero` and `Semantic Scholar` to enable metadata loading and external search.

## Installing Ollama
1. Download Ollama from [ollama.ai](https://ollama.ai) for your operating system (Windows, macOS, or Linux).
2. Run the installer and follow the setup wizard.
3. After installation, start the Ollama service:
   - **Windows/macOS**: Ollama runs as a background service automatically after installation.
   - **Linux**: Start Ollama with `ollama serve` or configure it as a systemd service.
4. Verify Ollama is running by checking `http://localhost:11434/api/tags` in your browser (should return available models).
5. Pull the required model (e.g., `mistral-nemo`):
   ```bash
   ollama pull mistral-nemo
   ```
6. Confirm the model is available with `ollama list`.

## Important files and entry points
1. `functionalTests/application_demo.ipynb` — interactive demo and evaluation notebook.
2. `backend/services/*` — core services: embedding, vector DB, retriever, evaluator, Semantic Scholar client.
3. `pdfProcessing/*` — PDF extraction and chunking utilities.
4. `llmAG/*` — RAG pipeline and LLM integration.
5. `backend/utils.py` — `query_rag`, ingestion helpers, logging and evaluation utilities.
6. `data/testPDFs` — example PDFs used for ingestion (local test data).
7. `outputs/application_demo` — demo outputs and saved evaluation results.

## Environment configuration
1. Create a `.env` file at the repo root and set values as needed:
   - `ZOTERO_LIBRARY_ID`
   - `ZOTERO_API_KEY`
   - `SEMANTIC_SCHOLAR_API_KEY`
   - `OLLAMA_BASE_URL` (default in notebook: `http://localhost:11434`)
2. Many runtime flags are configured in the demo notebook (e.g., `EMBEDDER_TYPE`, `CLEAR_DB_ON_RUN`, `MAX_CHUNK_SIZE`).

### Quick setup
```bash
# Install dependencies
pip install -r requirements.txt

# Optional: export/use a virtual environment
python -m venv .venv
source .venv/Scripts/activate     # on Windows (PowerShell: .venv\Scripts\Activate.ps1)
pip install -r requirements.txt
```

## Using conda (alternative)
If you prefer conda, create the environment from the provided environment.yaml:
```bash
conda env create -f environment.yaml
conda activate genai   # or the name specified in environment.yaml
```

```bash
# Unzip demo papers (required for the application demo)
unzip data/testPDFs.zip -d data/  # on Windows, use: expand-archive -path data/testPDFs.zip -destinationpath data/
```

## Running the demo
1. Ensure `OLLAMA` (or configured LLM) is running if using model `mistral-nemo`.
2. Populate `.env` with API keys if you want Zotero/ Semantic Scholar integration.
3. Open and run `functionalTests/application_demo.ipynb` in PyCharm or Jupyter. The notebook:
   - Initializes services and embedder
   - Optionally clears and re-populates the Chroma DB (`CLEAR_DB_ON_RUN`)
   - Demonstrates three RAG queries (tiered difficulty)
   - Runs systematic evaluation and saves results to `outputs/application_demo`

## Typical troubleshooting
1. LLM connection errors: confirm `OLLAMA` is running and `OLLAMA_BASE_URL` is correct.
2. No Zotero metadata: verify `ZOTERO_API_KEY` and `ZOTERO_LIBRARY_ID` are set in `.env`.
3. Semantic Scholar rate limits or empty results: check `SEMANTIC_SCHOLAR_API_KEY` and be aware of API limits.
4. Chroma DB path issues: ensure `./backend/chroma_db` is writable or adjust `CHROMA_PATH`.

## Evaluation
- The demo notebook runs the `EnhancedRAGEvaluator` end-to-end and writes `evaluation_results.csv` to `outputs/application_demo`.

## Project structure (high level)
1. `backend/` \- services, retriever, evaluator, vector DB adapter
2. `pdfProcessing/` \- PDF extraction and chunking utilities
3. `zotero_integration/` \- Zotero client and metadata loader
4. `llmAG/` \- LLM wrappers and RAG pipeline
5. `functionalTests/` \- demo notebooks and usage examples
6. `data/` \- sample PDFs and test data
7. `outputs/` \- demo outputs and saved artifacts
