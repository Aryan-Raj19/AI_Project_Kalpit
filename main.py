"""
AmbedkarGPT-Intern-Task - main.py
Simple command-line RAG Q&A using:
 - LangChain (new LCEL architecture)
 - ChromaDB for local vector store
 - HuggingFaceEmbeddings (sentence-transformers/all-MiniLM-L6-v2)
 - Ollama (Mistral 7B)

Requirements:
 - pip install langchain langchain-community langchain-openai langchain-text-splitters chromadb sentence-transformers
 - Ollama installed + mistral model pulled
"""

import os
from pathlib import Path

# Updated LangChain imports
from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.llms import Ollama
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import PromptTemplate


def build_or_load_vectorstore(
    speech_file: str,
    persist_directory: str = "chroma_db",
    chunk_size: int = 800,
    chunk_overlap: int = 100,
):
    persist_path = Path(persist_directory)
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

    # Load existing DB
    if persist_path.exists() and any(persist_path.iterdir()):
        print(f"Loading existing Chroma DB: {persist_directory}")
        vectordb = Chroma(persist_directory=persist_directory, embedding_function=embeddings)
        return vectordb

    print("Building new vector DB...")

    if not os.path.exists(speech_file):
        raise FileNotFoundError(f"speech.txt missing: {speech_file}")

    loader = TextLoader(speech_file, encoding="utf-8")
    docs = loader.load()

    splitter = CharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separator="\n",
        length_function=len,
    )

    split_docs = splitter.split_documents(docs)
    print(f"Created {len(split_docs)} text chunks.")

    vectordb = Chroma.from_documents(
        documents=split_docs,
        embedding=embeddings,
        persist_directory=persist_directory,
    )

    vectordb.persist()
    print(f"Saved vector DB to: {persist_directory}")

    return vectordb


def build_rag_chain(vectordb, ollama_model="mistral"):
    llm = Ollama(model=ollama_model)

    retriever = vectordb.as_retriever(search_kwargs={"k": 3})

    prompt = PromptTemplate(
        input_variables=["context", "question"],
        template=(
            "You are AmbedkarGPT. Answer the question strictly using only the context below.\n\n"
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer:"
        )
    )

    rag_chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
    )

    return rag_chain


def cli_loop(rag_chain):
    print("\nAmbedkarGPT Ready. Ask questions based ONLY on speech.txt.")
    print("Type 'quit' to exit.\n")

    while True:
        query = input("Q> ").strip()
        if query.lower() in ["quit", "exit"]:
            print("Goodbye.")
            break
        if not query:
            continue

        answer = rag_chain.invoke(query)
        print("\n--- Answer ---")
        print(answer.strip())
        print("---------------\n")


def main():
    root = Path(__file__).parent
    speech_file = root / "speech.txt"
    persist_dir = root / "chroma_db"

    vectordb = build_or_load_vectorstore(str(speech_file), str(persist_dir))
    rag_chain = build_rag_chain(vectordb, ollama_model="mistral")
    cli_loop(rag_chain)


if __name__ == "__main__":
    main()
