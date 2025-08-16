from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os

from services.pdf_processor import process_pdf
from services.embedder import embed_and_store
from config import UPLOAD_FOLDER

import pickle, faiss
from sentence_transformers import SentenceTransformer
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate


# LLM via LangChain

llm = ChatOpenAI(
    temperature=0.2, 
)

# Prompt template
prompt_template = """
You are a helpful assistant.
Use the following context from the PDF to answer the question.
If the answer is not in the context, say "I don't know based on the provided document."

Context:
{context}

Question: {question}
Answer:
"""
prompt = PromptTemplate(
    template=prompt_template, 
    input_variables=["context", "question"]
)

model = SentenceTransformer('all-MiniLM-L6-v2')
app = Flask(__name__)
CORS(app) 

app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024

ALLOWED_EXTENSIONS = {"pdf"}
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

# Upload PDF
@app.route("/upload", methods=["POST"])
def upload_pdf():
    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    # make filename safe and save
    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        chunks = process_pdf(filepath)
        index_path = embed_and_store(chunks, filename.rsplit(".", 1)[0])
        print('File embedded successfully')
        return jsonify({"message": "File embedded successfully", "index_path": index_path}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Ask question
@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json()
        question = data.get("question", "").strip()
        if not question:
            return jsonify({"error": "Question is required"}), 400

        question_embedding = model.encode([question])

        base_dir = os.path.dirname(os.path.abspath(__file__))
        faiss_path = os.path.join(base_dir, "vector_store", "Seize_2023", "index.faiss")
        pkl_path = os.path.join(base_dir, "vector_store", "Seize_2023", "index.pkl")
        faiss_index = faiss.read_index(faiss_path)

        with open(pkl_path, "rb") as f:
            docstore, id_map = pickle.load(f)

        k = 3
        distances, indices = faiss_index.search(question_embedding, k)

        retrieved_chunks = []
        for idx in indices[0]:
            if idx != -1:
                doc_id = id_map[idx]
                retrieved_chunks.append(docstore._dict[doc_id].page_content)

        context_text = "\n\n".join(retrieved_chunks)

        formatted_prompt = prompt.format(context=context_text, question=question)
        llm_response = llm.invoke(formatted_prompt)

        return jsonify({
            "question": question,
            "retrieved_chunks": retrieved_chunks,
            "answer": llm_response.content
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(debug=True)
