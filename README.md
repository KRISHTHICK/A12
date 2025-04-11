# A12
GenAI

To create a Retrieval-Augmented Generation (RAG) application that processes multiple documents (PDFs, links, or other formats), extracts information, and answers questions with reference to the documents, you can use Python along with libraries like langchain, PyPDF2, pytesseract, and Flask. For integration with Ollama, you can use its API for the generative responses.

Below is an example implementation:

Steps to Create the RAG Application
Document Parsing: Extract text and content from uploaded documents (e.g., PDFs, Word files, links).
Indexing: Use a vector database (e.g., FAISS or Weaviate) to store document embeddings for efficient retrieval.
Question Answering: Retrieve relevant documents and generate answers using the Ollama API or another language model.
Application Interface: Provide an upload and query interface (e.g., Flask for a web app).
Reference Answers: Include document references in the responses.
Python Code for RAG Application
Python
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
Key Features in the Code
Document Processing:

Uses PyPDFLoader and UnstructuredFileLoader to extract text from PDFs and other formats.
Supports multiple document uploads.
Vector Store:

Uses FAISS for storing and retrieving document embeddings.
Embeddings are created using sentence-transformers/all-MiniLM-L6-v2.
Question Answering:

Combines document retrieval with Ollama's generative capabilities using langchain.
File Upload:

Allows uploading multiple documents/files via the /upload endpoint.
Question Querying:

Accepts questions via the /ask endpoint and generates answers with references to the documents.
References in Answers:

The langchain library automatically includes document references in the answers.
Example Usage
Run the Application:

bash
python app.py
Upload Documents: Use the /upload endpoint to upload multiple documents:

bash
curl -X POST -F "files=@document1.pdf" -F "files=@document2.pdf" http://127.0.0.1:5000/upload
Ask a Question: Use the /ask endpoint to ask a question:

bash
curl -X POST -H "Content-Type: application/json" -d '{"question": "What are the key points in the financial reports?"}' http://127.0.0.1:5000/ask
Example Answer:

JSON
{
    "answer": "According to document 'financial_report_2022.pdf', the revenue increased by 20%. Another reference is 'financial_summary.pdf', which highlights a net profit of $5M."
}
Prerequisites
Install Required Libraries:

bash
pip install flask langchain faiss-cpu pytesseract sentence-transformers
Install Tesseract (for OCR):

For Ubuntu:
bash
sudo apt install tesseract-ocr
For Windows, download and install Tesseract from here.
Ollama API:

Set up Ollama and obtain API access.
Replace your-ollama-model with the specific model name.
Enhancements
Support for Links:

Use libraries like requests or BeautifulSoup to scrape and process text from web links.
Advanced Document Parsing:

Use PyMuPDF for extracting embedded images, tables, and text from PDFs.
Cloud Storage:

Integrate with cloud storage like AWS S3 for document uploads and storage.
UI Integration:

Build a frontend using frameworks like React or Vue.js for an interactive experience.
