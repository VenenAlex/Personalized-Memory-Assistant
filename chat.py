import os
import time
from pathlib import Path

# -------------------- CONFIGURE GEMINI API KEY --------------------
# ✅ Replace with your actual API key:
os.environ["GOOGLE_API_KEY"] = "YOUR_GOOGLE_API_KEY_HERE"

import google.generativeai as genai
genai.configure(api_key=os.environ["GOOGLE_API_KEY"])

# -------------------- IMPORTS --------------------
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain_google_genai import ChatGoogleGenerativeAI

# -------------------- SETTINGS --------------------
DOCUMENTS_DIR = "./new_articles"  # Folder containing .txt files
CHROMA_PERSIST_DIR = "./chroma_persist"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200
TOP_K = 2
MODEL_NAME = "gemini-2.0-flash"

# -------------------- 1. LOAD DOCUMENTS --------------------
def load_documents(directory):
    loader = DirectoryLoader(
        directory,
        glob="./*.txt",
        loader_cls=TextLoader
    )
    docs = loader.load()
    return docs

# -------------------- 2. SPLIT INTO CHUNKS --------------------
def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP
    )
    return splitter.split_documents(documents)

# -------------------- 3. BUILD / LOAD CHROMA DB --------------------
def build_or_load_chroma(docs):
    embedding = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    if not Path(CHROMA_PERSIST_DIR).exists():
        vectordb = Chroma.from_documents(
            documents=docs,
            embedding=embedding,
            persist_directory=CHROMA_PERSIST_DIR
        )
        vectordb.persist()
    else:
        vectordb = Chroma(
            persist_directory=CHROMA_PERSIST_DIR,
            embedding_function=embedding
        )
    return vectordb

# -------------------- 4. INITIALIZE LLM --------------------
def init_gemini():
    return ChatGoogleGenerativeAI(
        model=MODEL_NAME,
        google_api_key=os.environ["GOOGLE_API_KEY"]
    )

# -------------------- 5. CREATE QA CHAIN --------------------
def create_qa_chain(llm, retriever):
    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )

# -------------------- 6. PRINT RESPONSE --------------------
def print_response(response):
    print("\n=== ANSWER ===\n")
    print(response["result"])
    print("\n=== SOURCES ===\n")
    sources = response.get("source_documents", [])
    if not sources:
        print("No sources returned.")
    else:
        for doc in sources:
            print(doc.metadata.get("source", "No source field"))


# -------------------- MAIN EXECUTION --------------------
def main():
    if not Path(DOCUMENTS_DIR).exists():
        raise FileNotFoundError(
            f"Directory '{DOCUMENTS_DIR}' not found. Place your .txt files there."
        )

    print("Loading documents...")
    documents = load_documents(DOCUMENTS_DIR)
    print(f"{len(documents)} documents loaded.")

    print("Splitting documents...")
    chunks = split_documents(documents)
    print(f"{len(chunks)} chunks created.")

    print("Building / Loading Chroma DB...")
    vectordb = build_or_load_chroma(chunks)
    retriever = vectordb.as_retriever(search_kwargs={"k": TOP_K})

    print("Initializing Gemini LLM...")
    llm = init_gemini()

    print("Building RetrievalQA chain...")
    qa_chain = create_qa_chain(llm, retriever)

    # Example query
    query = "How much money did Microsoft raise?"
    print(f"\nQuery: {query}")
    response = qa_chain(query)
    print_response(response)

    # Optional: Manual input loop
    while True:
        user_query = input("\nEnter your question (or 'exit'): ").strip()
        if user_query.lower() == "exit":
            print("Goodbye!")
            break
        response = qa_chain(user_query)
        print_response(response)


if __name__ == "__main__":
    main()
