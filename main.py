
import os
import json
import pytesseract
from flask import Flask, request, jsonify
from langchain.document_loaders import PyPDFLoader, UnstructuredFileLoader
from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.chains import RetrievalQA
from langchain.llms import Ollama

# Initialize Flask app
app = Flask(__name__)

# Initialize vector database and embeddings
embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
vector_store = FAISS(embedding_model.embed_query, embedding_model)

# Ollama API initialization
ollama_model = Ollama(model_name="your-ollama-model")

# Upload folder
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def process_file(file_path):
    """Process and extract text from the uploaded file."""
    if file_path.endswith(".pdf"):
        loader = PyPDFLoader(file_path)
    else:
        loader = UnstructuredFileLoader(file_path)
    documents = loader.load()
    return documents

@app.route('/upload', methods=['POST'])
def upload_documents():
    """Upload and process documents."""
    if 'files' not in request.files:
        return jsonify({"error": "No files uploaded"}), 400
    
    files = request.files.getlist('files')
    all_documents = []
    for file in files:
        file_path = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(file_path)
        # Process each file
        documents = process_file(file_path)
        all_documents.extend(documents)
    
    # Add documents to the vector store
    vector_store.add_documents(all_documents)
    return jsonify({"message": f"{len(files)} files processed and indexed successfully."}), 200

@app.route('/ask', methods=['POST'])
def ask_question():
    """Answer a question based on the uploaded documents."""
    data = request.get_json()
    question = data.get("question")
    if not question:
        return jsonify({"error": "No question provided"}), 400
    
    # Create a RetrievalQA chain
    retriever = vector_store.as_retriever(search_kwargs={"k": 5})  # Top 5 results
    qa_chain = RetrievalQA.from_chain_type(llm=ollama_model, retriever=retriever)
    
    # Get the answer
    result = qa_chain.run(question)
    return jsonify({"answer": result}), 200

if __name__ == "__main__":
    app.run(debug=True)
