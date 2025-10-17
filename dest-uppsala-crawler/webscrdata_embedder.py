from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings  
from langchain_community.document_loaders import JSONLoader
from langchain_chroma import Chroma
import os, time
from dotenv import load_dotenv
from langchain_google_genai._common import GoogleGenerativeAIError
load_dotenv() # should make a .env file, and store GEMINI_API_KEY=AIe4...
# from sentence_transformers import SentenceTransformer
from langchain_huggingface.embeddings import HuggingFaceEmbeddings

## This code is to be run only once or possibly, minimal number of times (as embedding rate limits are finite).

# https://python.langchain.com/docs/integrations/document_loaders/json/#extracting-metadata
def webscr_metadata_extractor(record, metadata):
    metadata["title"] = record.get("title") # These are also important fields that should be added
    metadata["url"] = record.get("url") # can help in website-based RAG later
    return metadata

vector_store = Chroma(
    persist_directory='./webscr_chunks_chromadb',
    embedding_function=HuggingFaceEmbeddings(model_name="google/embeddinggemma-300m")
    # GoogleGenerativeAIEmbeddings(
    #     model="models/gemini-embedding-001", 
    #     google_api_key=os.getenv('GEMINI_API_KEY')
    # )
)
webscraped='./uppsala_chunks.json'
loader = JSONLoader(
    file_path=webscraped,
    jq_schema='.[]',
    content_key="text",
    metadata_func=webscr_metadata_extractor
)
docs = loader.load()
en_docs = [doc for doc in docs if "https://destinationuppsala.se/en" in doc.metadata['url']]

num_batches = 445

for i in range(1,num_batches): #
    start = int((i-1)*(len(docs)//(num_batches-1)))
    end = int(i*(len(docs)//(num_batches-1)))
    
    vector_store.add_documents(documents=docs[start:end])
    print("Embedded from", start, ":", end, ">> on batch:", i, "of", num_batches-1)
    #time.sleep(61)