# GenAI

# 1. Notebook only

## Setup

Preferable Python Version: 3.11.14

Follow these steps to get your development environment running.

### 1. Install PyTorch Optional if Docker is used

PyTorch installation varies based on your Operating System and CUDA (Graphics Card) version.

* Visit the official [PyTorch Get Started](https://pytorch.org/get-started/locally/) page.
* Select your preferences (OS, Package, Compute Platform).
* Run the generated install command.

### 2. Install Dependencies

Choose **one** of the following methods to install the remaining requirements.

**Option A: Using Pip**

```bash
pip install -r requirements.txt
```

**Option B: Using Conda**

```bash
conda env create -f environment.yml
conda activate genai_env
```

### 3. Data Preparation

To run the test suite, you must set up the local data directory.

1. Create a folder named testPDFs inside a data directory.
2. Populate it with sample PDF files for testing.

Command Line Quick Setup:

```bash
mkdir -p data/testPDFs
```

**Expected Structure:**

```
├── data/ 
│   └── testPDFs/ 
│       ├── document1.pdf
│       └── document2.pdf
├── requirements.txt
| ...
```

### 4. Run Notebook

Go to `backend/pipelineTest.ipynb` and run the notebook.
Running it for first time will take some time because the models have to be downloaded.

# 2. Backend API 

The project includes a FastAPI backend that serves the RAG pipeline. You can run this either locally on your machine or
inside a Docker container.

## Setup
## 0. Add `.env` file to root folder
Create a `.env` file in the root folder of the project with the following content:

```
SEMANTIC_SCHOLAR_API_KEY=your_openai_api_key_here
```
## 1. Install Docker (if not already installed)

Follow the instructions on the official [Docker Installation Guide](https://docs.docker.com/get-docker/) to install
Docker on your system.

## 2a. Run Locally

Follow step 1, 2 and 3 from notebook to get the backend running

1. **Start the Server:**
   Run the following command from the project root:

```bash
uvicorn backend.main:app --host 0.0.0.0 --port 8000 --reload
```

2. **Access the API Documentation:**
   Open your web browser and navigate to `http://localhost:8000/docs` to access the interactive API documentation.

- Swagger UI: http://localhost:8000/docs
- ReDoc: http://localhost:8000/redoc

## 2b. Running in Docker (Easy method)

We provide helper scripts to automatically handle GPU detection, volume mounting (for hot-reloading code), and
cross-platform compatibility.\
**On Windows (PowerShell):**

```powershell
.\run_container.ps1
```

**On Linux / macOS / Git Bash:**

```bash
./run_container.sh
```

\
**Common Flags:**

- **Force Rebuild:** Use if you changed `requirements.txt` or the `Dockerfile`.
    - PowerShell: `.\run_container.ps1 -Rebuild`
    - Bash: `./run_container.sh -Rebuild`

\
The container will automatically mount your source code, so changes you make in backend/ will trigger a server
restart (Hot Reload) without needing to rebuild the image.