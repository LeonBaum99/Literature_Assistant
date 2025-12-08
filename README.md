# GenAI
## 🛠Setup
Follow these steps to get your development environment running.
### 1. Install PyTorch
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