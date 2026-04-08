from flask import Flask, request, jsonify
from flask_cors import CORS
from werkzeug.utils import secure_filename
import os

from langchain_openai import ChatOpenAI
try:
    from langchain_core.prompts import PromptTemplate
except ImportError:
    from langchain.prompts import PromptTemplate

from services.pdf_processor import process_pdf
from services.embedder import embed_and_store
from services.hybrid_retriever import hybrid_retrieve, list_available_indexes
from config import (
    UPLOAD_FOLDER,
    DENSE_TOP_K,
    SPARSE_TOP_K,
    FINAL_TOP_K,
    HYBRID_ALPHA,
)

from dotenv import load_dotenv

load_dotenv()


def build_llm(model_provider, model_name, api_key):
    provider = (model_provider or "openai").strip().lower()
    if provider != "openai":
        raise ValueError("Only OpenAI model provider is supported right now.")

    if not api_key:
        raise ValueError("OpenAI API key is required for OpenAI model selection.")

    llm_kwargs = {
        "temperature": 0.2,
        "api_key": api_key,
    }
    if model_name:
        llm_kwargs["model"] = model_name

    return ChatOpenAI(**llm_kwargs)

prompt_template = """
You are a helpful assistant.
Use only the following context from the uploaded PDF to answer the question.
If the answer is not in the context, say "I don't know based on the provided document."

Context:
{context}

Question: {question}
Answer:
"""
prompt = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

app = Flask(__name__)
CORS(app)
app.config["MAX_CONTENT_LENGTH"] = 100 * 1024 * 1024

ALLOWED_EXTENSIONS = {"pdf"}
ACTIVE_INDEX_NAME = None


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def resolve_index_name(requested_index_name=None):
    if requested_index_name:
        return requested_index_name

    if ACTIVE_INDEX_NAME:
        return ACTIVE_INDEX_NAME

    available = list_available_indexes()
    return available[0] if available else None


@app.route("/upload", methods=["POST"])
def upload_pdf():
    global ACTIVE_INDEX_NAME

    if "file" not in request.files:
        return jsonify({"error": "No file provided"}), 400

    file = request.files["file"]
    if file.filename == "":
        return jsonify({"error": "Empty filename"}), 400

    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type"}), 400

    filename = secure_filename(file.filename)
    filepath = os.path.join(UPLOAD_FOLDER, filename)
    file.save(filepath)

    try:
        chunks = process_pdf(filepath)
        index_name = filename.rsplit(".", 1)[0]
        store_info = embed_and_store(chunks, index_name)

        ACTIVE_INDEX_NAME = index_name

        return jsonify(
            {
                "message": "File embedded successfully",
                "index_name": index_name,
                "collection_name": store_info["collection_name"],
                "vector_store_path": store_info["persist_directory"],
            }
        ), 200
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/indexes", methods=["GET"])
def list_indexes():
    return jsonify(
        {
            "active_index_name": ACTIVE_INDEX_NAME,
            "indexes": list_available_indexes(),
        }
    )


@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json(silent=True) or {}
        question = data.get("question", "").strip()
        if not question:
            return jsonify({"error": "Question is required"}), 400

        model_provider = data.get("model_provider", "openai")
        model_name = data.get("model_name", "gpt-4o-mini")
        api_key = (data.get("api_key") or "").strip()

        requested_index_name = data.get("index_name") or data.get("collection_name")
        index_name = resolve_index_name(requested_index_name)
        if not index_name:
            return jsonify({"error": "No uploaded document found. Upload a PDF first."}), 400

        ranked_results = hybrid_retrieve(
            query=question,
            index_name=index_name,
            dense_k=DENSE_TOP_K,
            sparse_k=SPARSE_TOP_K,
            final_k=FINAL_TOP_K,
            alpha=HYBRID_ALPHA,
        )

        retrieved_chunks = [item["content"] for item in ranked_results]
        context_text = "\n\n".join(retrieved_chunks) if retrieved_chunks else "No relevant context found."

        formatted_prompt = prompt.format(context=context_text, question=question)
        llm = build_llm(
            model_provider=model_provider,
            model_name=model_name,
            api_key=api_key,
        )
        llm_response = llm.invoke(formatted_prompt)

        return jsonify(
            {
                "question": question,
                "index_name": index_name,
                "model_provider": model_provider,
                "model_name": model_name,
                "retrieved_chunks": retrieved_chunks,
                "retrieval_details": ranked_results,
                "answer": llm_response.content,
            }
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    app.run(debug=True, use_reloader=False)
