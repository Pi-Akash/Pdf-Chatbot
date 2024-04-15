"""
Steps to perform before running this application:
    1. Download and install Ollama from https://ollama.com/
    2. Install all the packages required as shown below.
    3. There might be some dependency packages which needs to be installed like pdfminer, chromadb, langchain-text-splitters etc. keep looking for package errors.
    4. We are using Nomic-Embed-text model from Ollama, follow the below instruction to download model into local
        # to download the nomic embed text model into local
        # !ollama pull nomic-embed-text
    5. Similarly, Download the Gemma 2b model from Ollama.
        # to download the gemma:2b model into local
        # !ollama pull gemma:2b

On the above system configuration, the model takes a while to provide responses.
"""

# imports for program

# streamlit imports for UI
import streamlit as st
from streamlit_chat import message
import os

# UnstructuredPDFLoader to read pdf files
from langchain_community.document_loaders import UnstructuredPDFLoader
# OllamaEmbeddings to generate local embedding
from langchain_community.embeddings import OllamaEmbeddings
# RecursiveCharacterTextSplitter to split and chunk text
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Chroma to connect with ChromaDB
from langchain_community.vectorstores import Chroma
# ConversationBufferMemory to keep conversations in buffer memory
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
# ChatOllama to chat with local model
from langchain_community.chat_models import ChatOllama
# to keep track of local embeddings
import sqlite3

def sqlite_connection(db_name):
    """
    The function creates connection to local sqlite3 db and returns the connection
    """
    # creates connection with sqlite3 db
    connection = sqlite3.connect(db_name)
    return connection

def get_collection_id(collection_name):
    """
    The function returns collection_id from the local sqlite3 database 
    """
    try:
        # create connection
        conn = sqlite_connection(db_name = "Embeddings/chroma.sqlite3")
    except Exception as e:
        print(e)
    
    with conn:
        cursor = conn.cursor()
        try:
            rows = cursor.execute(
                """
                with temp_table as (
                    select 
                    s.id, 
                    s.collection, 
                    c.name 
                    from segments as s inner join collections as c 
                    on s.collection = c.id where scope = "VECTOR"
                )
            
                select id from temp_table where name = ?
                """, (collection_name,)
                ).fetchall()
        except:
            return None
    
    if len(rows) == 0:
        return None

    return rows[0][0]
        

def get_pdf_text(pdf_doc_filename):
    """
    The function takes in the name of the document and returns the text from the document.
    """
    # initialize empty text variable
    text = ""
    try:
        # store text from pdf in text
        pdf_loader = UnstructuredPDFLoader(file_path = os.path.join("Documents", pdf_doc_filename))
        text = pdf_loader.load()
    except Exception as e:
        text = e
    return text

def get_text_chunks(pdf_data):
    """
    The function reads the pdf text data and returns chunks of texts
    Reversive Character Text Splitter overlaps chunks between the document chunks.
    """
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size = 1000, 
        chunk_overlap = 50,
        length_function = len
        )
    chunks = text_splitter.split_documents(pdf_data)
    return chunks

def get_hash(text):
    """
    Function generates a simple hash based of filename to differentiate between embeddings
    Helps in memoization of embeddings
    """
    stripped_text = text.strip().replace(" ", "")
    even_characters = stripped_text[::2]
    reverse_characters = stripped_text[::-1]
    combined = even_characters + "_" + reverse_characters
    return combined
    
def check_collection_exists(pdf_doc):
    """
    The function checks if a collection already exists for a document.
    """
    # create document hash
    document_hash_name = get_hash(pdf_doc)
    # generate collection name
    collection_name = "Embedding_{}".format(document_hash_name)
    # get collection id
    collection_id = get_collection_id(collection_name)
    if collection_id in os.listdir("Embeddings"):
        #print("Embedding Exists")
        return True
    else:
        #print("Embedding does not exist")
        return False

def get_vectorstore(text_chunks, pdf_doc):
    """
    The function takes in text chunks and converts them into embeddings using nomic-embed-text model from ollama.
    And, store the embeddings into ChromaDB and returns those embeddings
    """    
    #st.write(check_collection_exists(pdf_doc))
    document_hash_name = get_hash(pdf_doc)
    if check_collection_exists(pdf_doc):
        # load collection from disk
        vector_db = Chroma(
            persist_directory = "./Embeddings",
            collection_name = "Embedding_{}".format(document_hash_name),
            embedding_function = OllamaEmbeddings(model = "nomic-embed-text", show_progress = True)
            )
    else:
        # create new collection in local
        vector_db = Chroma.from_documents(
            documents = text_chunks,
            embedding = OllamaEmbeddings(model = "nomic-embed-text", show_progress = True),
            collection_name = "Embedding_{}".format(document_hash_name),
            persist_directory = "./Embeddings"
        )
    st.caption("Embeddings Loaded in Memory")
    return vector_db

def get_conversation_chain(vectorstore):
    """
    The function takes in the embeddings to retrieve answers and store the conversation in memory.
    """
    # load local model
    local_model = "gemma:2b"
    llm = ChatOllama(model = local_model)
    # create buffer memory object
    conv_memory = ConversationBufferMemory(memory_key = "chat_history", return_messages = True)
    # create conversation chain using local llm and embeddings in Chroma DB as retriever, store them in memory
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm = llm,
        retriever = vectorstore.as_retriever(),
        memory = conv_memory
    )
    return conversation_chain

def handle_user_input(user_input):
    """
    Function handles user input, generates response and displays chat history with document.
    """
    response = st.session_state.conversation({"question" : user_input})
    #st.write(response)
    st.session_state.chat_history = response["chat_history"]
     
    for i, msg in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            message(msg.content, is_user = True)
        else:
            message(msg.content, is_user = False)
             

def main():
    
    # set page configuration
    st.set_page_config(
        page_title = "PDF-Bot",
        page_icon = ":robot_face:"
    )

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    
    st.header("Chat with Documents")
    user_input = st.text_input("Ask your question here...", key = "user_input")
    
    if user_input:
        handle_user_input(user_input)
    
    with st.sidebar:
        st.subheader("Your Pdf File: ")
        
        pdf_doc_filename = st.selectbox(
            label = "Select a document from the list and click on the 'Process'",
            options = os.listdir("Documents")
        )
        process_button = st.button("Process")
        
        if process_button:
            with st.spinner("Processing"):
                # get the pdf text
                pdf_data = get_pdf_text(pdf_doc_filename)
                #st.write(pdf_data)
                
                # get the text chunks
                text_chunks = get_text_chunks(pdf_data)
                #st.write(text_chunks)
                
                # create vector store
                vectorstore = get_vectorstore(text_chunks, pdf_doc_filename)
                #st.write(vectorstore)
                
                # conversation chain
                st.session_state.conversation = get_conversation_chain(vectorstore)
          
if __name__ == "__main__":
    main()