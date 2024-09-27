import os
import streamlit as st
import time
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.schema import Document  # Import Document class

from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env (especially openai api key)

st.title("JenkinsInsight: Build Issue Analyzer 📈")
st.sidebar.title("Upload Text Files")

# Allow users to upload .txt files
uploaded_files = st.sidebar.file_uploader("Choose text files", type=["txt"], accept_multiple_files=True)

process_files_clicked = st.sidebar.button("Process Files")
index_path = "faiss_store_openai"

main_placeholder = st.empty()
llm = OpenAI(temperature=0.9, max_tokens=500)

# Initialize embeddings outside of the conditional block
embeddings = OpenAIEmbeddings()

if process_files_clicked:
    if uploaded_files:
        # Read the contents of the uploaded files and wrap them in Document objects
        documents = []
        for uploaded_file in uploaded_files:
            file_content = uploaded_file.read().decode("utf-8")  # Read and decode text content
            # Create Document objects with metadata (source: file name)
            documents.append(Document(page_content=file_content, metadata={"source": uploaded_file.name}))
        
        main_placeholder.text("File loading and processing... Started...✅✅✅")

        # Split data
        text_splitter = RecursiveCharacterTextSplitter(
            separators=['\n\n', '\n', '.', ','],
            chunk_size=1000
        )
        main_placeholder.text("Text Splitting... Started...✅✅✅")
        docs = text_splitter.split_documents(documents)

        # Create embeddings and save them to FAISS index
        vectorstore_openai = FAISS.from_documents(docs, embeddings)
        main_placeholder.text("Embedding Vector Started Building...✅✅✅")
        time.sleep(2)

        # Save the FAISS index to a file
        vectorstore_openai.save_local(index_path)

    else:
        main_placeholder.text("No files were uploaded.")

query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(index_path):
        # Load the FAISS index from the file with dangerous deserialization allowed
        vectorstore = FAISS.load_local(index_path, embeddings, allow_dangerous_deserialization=True)
        chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
        result = chain({"question": query}, return_only_outputs=True)

        # Display the answer
        st.header("Answer")
        st.write(result["answer"])

        # Display sources, if available
        sources = result.get("sources", "")
        if sources:
            st.subheader("Sources:")
            sources_list = sources.split("\n")  # Split the sources by newline
            for source in sources_list:
                st.write(source)
