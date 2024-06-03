import os
import streamlit as st
import pickle
import time
from langchain import OpenAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.document_loaders import UnstructuredURLLoader
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS

from dotenv import load_dotenv
load_dotenv()  # take environment variables from .env (especially openai api key)

st.title("RockyBot: News Research Tool 📈")
st.sidebar.title("News Article URLs")

urls = []
for i in range(3):
    url = st.sidebar.text_input(f"URL {i+1}")
    urls.append(url)

process_url_clicked = st.sidebar.button("Process URLs")
file_path = "faiss_store_openai.pkl"

main_placeholder = st.empty()
llm = OpenAI(temperature=0.9, max_tokens=500)

if process_url_clicked:
    # load data
    loader = UnstructuredURLLoader(urls=urls)
    main_placeholder.text("Data Loading...Started...✅✅✅")
    data = loader.load()
    # split data
    text_splitter = RecursiveCharacterTextSplitter(
        separators=['\n\n', '\n', '.', ','],
        chunk_size=1000
    )
    main_placeholder.text("Text Splitter...Started...✅✅✅")
    docs = text_splitter.split_documents(data)
    # create embeddings and save it to FAISS index
    embeddings = OpenAIEmbeddings()
    vectorstore_openai = FAISS.from_documents(docs, embeddings)
    main_placeholder.text("Embedding Vector Started Building...✅✅✅")
    time.sleep(2)
	

    # Save the FAISS index to a pickle file
    with open(file_path, "wb") as f:
        pickle.dump(vectorstore_openai, f)

query = main_placeholder.text_input("Question: ")
if query:
    if os.path.exists(file_path):
        with open(file_path, "rb") as f:
            vectorstore = pickle.load(f)
            chain = RetrievalQAWithSourcesChain.from_llm(llm=llm, retriever=vectorstore.as_retriever())
            result = chain({"question": query}, return_only_outputs=True)
            # result will be a dictionary of this format --> {"answer": "", "sources": [] }
            st.header("Answer")
            st.write(result["answer"])

            # Display sources, if available
            sources = result.get("sources", "")
            if sources:
                st.subheader("Sources:")
                sources_list = sources.split("\n")  # Split the sources by newline
                for source in sources_list:
                    st.write(source)

#################################################################
"""
This Python code implements a "RockyBot: News Research Tool" using Streamlit  for a user interface. Here's a breakdown of what the code does:

1. Imports and Setup:

Imports necessary libraries like streamlit, langchain, pickle, and OpenAI.
Loads environment variables (likely including an OpenAI API key) using dotenv.
Sets the Streamlit app title and creates a sidebar for user input.
2. User Input for URLs:

Creates text input fields in the sidebar for users to enter up to 3 news article URLs.
Stores these URLs in a list called urls.
Provides a button labeled "Process URLs" to trigger data processing.
3. Data Processing (Triggered by Button Click):

Loads the article data from the provided URLs using UnstructuredURLLoader.
Splits the loaded data into smaller chunks using RecursiveCharacterTextSplitter.
Creates vector representations (embeddings) for each chunk using OpenAIEmbeddings.
Builds a FAISS vectorstore to efficiently search the embeddings.
Saves the FAISS vectorstore as a pickle file (faiss_store_openai.pkl) for later use.
4. User Input for Question:

Creates a text input field in the main app window for users to enter their question.
5. Answering the Question (if a question is provided and the pickle file exists):

Loads the saved FAISS vectorstore from the pickle file.
Creates a retrieval chain using RetrievalQAWithSourcesChain that combines the OpenAI language model (llm) for answering questions and the FAISS vectorstore for retrieving relevant article snippets.
Processes the user's question through the retrieval chain and retrieves the answer and source information.
Displays the answer in a header element.
If sources are available, it splits them by newline and displays them in a list under a subheader.
Overall, this code allows users to input news article URLs, process the articles, and then ask questions about the content. The tool uses the OpenAI API and text embeddings to find relevant information and answer the user's questions.


"""

