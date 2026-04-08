# PDF Question Answering (Hybrid RAG)

A full-stack application that allows users to upload PDF documents and ask natural language questions about their content. It utilizes a **Hybrid Retrieval-Augmented Generation (RAG)** approach, combining both dense semantic search and sparse lexical (keyword) search to provide highly accurate answers based strictly on the uploaded document.

## Features
* **PDF Upload & Processing**: Seamlessly ingests, chunks, and locally embeds PDF content.
* **Hybrid RAG Engine**: Uses Hugging Face embeddings (dense) + BM25 (sparse) to fetch the most relevant context.
* **Local Vector Storage**: Uses ChromaDB to persist data locally without relying on paid cloud databases.
* **OpenAI Integration**: Synthesizes final answers securely using `gpt-4o-mini` (or your chosen model).
* **Modern Frontend**: Built with React for an intuitive chat interface.

## 🛠️ Tech Stack
* **Backend**: Python, Flask, LangChain, ChromaDB, HuggingFace Transformers
* **Frontend**: React.js, TailwindCSS (or Vanilla CSS)
* **LLM Provider**: OpenAI

---

## Getting Started

### 1. Setup Backend (Python)
Navigate to the root directory and create a `.env` file with your API key:
```env
OPENAI_API_KEY=your_openai_api_key_here
```

Install dependencies:
```bash
python -m venv .venv
source .venv/bin/activate  # (On Windows use .venv\Scripts\Activate.ps1)
pip install flask flask-cors python-dotenv langchain langchain-openai langchain-community langchain-text-splitters chromadb sentence-transformers pypdf langchain-huggingface
```

Run the API Server:
```bash
python app.py
```
*(The server will start on `http://127.0.0.1:5000`)*


### 2. Setup Frontend (React)
Open a new terminal tab and navigate into the deeply nested frontend folder:
```bash
cd frontend/frontend
```

Install Node dependencies and start the React server:
```bash
npm install
npm start
```
*(The web application will open on `http://localhost:3000`)*

---

## Usage
1. Open the React web interface in your browser.
2. Select a PDF file from your computer using the **Upload** button.
3. Wait for the success message to verify it has been locally embedded & saved effectively.
4. Start asking questions directly related to the document's content!
