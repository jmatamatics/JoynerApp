from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_openai import ChatOpenAI
from langchain_community.vectorstores import FAISS
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from dotenv import load_dotenv
import streamlit as st
from streamlit_pdf_viewer import pdf_viewer
import os
memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True, output_key='answer')

from langchain_community.document_loaders import PyMuPDFLoader
#import pytesseract
from PIL import Image
import fitz  # PyMuPDF
def image_to_text(file_path):
    text =''
    doc = fitz.open(file_path)
    for page in doc:
        pix = page.get_pixmap()
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        text += pytesseract.image_to_string(img)
        #doc.close()
    return text

def load_pdfs(pdf_docs):
    folder_path = 'data'
    os.makedirs(folder_path, exist_ok=True)
    #documents=''
    for pdf in pdf_docs:
        file_path =os.path.join(folder_path, pdf.name)
        with open(file_path, "wb") as f:
            f.write(pdf.getbuffer())
        #documents += image_to_text(file_path)       
        #documents = documents.extend(text)


    loader=PyPDFDirectoryLoader("./data") #, extract_images=True)
    documents = loader.load()
    #loader=PyMuPDFLoader(file_path, extract_images =True)    
    #documents = documents.extend(loader.load())
    return  documents


def chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=2000, chunk_overlap=200)
    #chunks = text_splitter.split_text(text)
    chunks = text_splitter.split_documents(text)
    return chunks



def index(text_chunks):
    embeddings = OpenAIEmbeddings()
    #vector = FAISS.from_texts(text_chunks, embeddings)
    vector = FAISS.from_documents(text_chunks, embeddings)
    vector.save_local("faiss_index")


def add_(text_chunks):
    embeddings = OpenAIEmbeddings()
    vector = FAISS.load_local("faiss_index", OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    vector.add_documents(text_chunks)
    vector.save_local("faiss_index")


def pdf_gpt(human_input):
    llm = ChatOpenAI(model='gpt-4')
    embeddings = OpenAIEmbeddings()
    vector = FAISS.load_local("faiss_index", embeddings, allow_dangerous_deserialization=True)
    retriever = vector.as_retriever()
    rel_docs= vector.similarity_search(human_input, k=4)
    gpt = ConversationalRetrievalChain.from_llm(
        llm, retriever, memory=memory, verbose=True,return_source_documents=False
    )
    result = gpt.invoke({"question": human_input})
    return result["answer"], rel_docs   




c ="""
#import pyheif
#from PIL import Image

def convert_heif_to_jpeg(heif_path, output_path):
    # Read HEIF file
    heif_file = pyheif.read(heif_path)
    
    # Convert to Pillow Image
    image = Image.frombytes(
        heif_file.mode, 
        heif_file.size, 
        heif_file.data,
        "raw",
        heif_file.mode,
        heif_file.stride,
    )
    
    # Save as JPEG
    image.save(output_path, "JPEG")

# Usage example
convert_heif_to_jpeg("input.heic", "output.jpg")
"""