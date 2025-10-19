## This code is to be run only once or possibly, minimal number of times (as embedding rate limits are finite).
# https://python.langchain.com/docs/integrations/document_loaders/json/#extracting-metadata

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings  
from langchain_community.document_loaders import JSONLoader
import os, time
from dotenv import load_dotenv
from langchain_google_genai._common import GoogleGenerativeAIError
load_dotenv() # should make a .env file, and store GEMINI_API_KEY=AIe4...
# from sentence_transformers import SentenceTransformer
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
embedding = HuggingFaceEmbeddings(model_name="google/embeddinggemma-300m")

# from langchain_chroma import Chroma
# vector_store = Chroma(
#     persist_directory='./webscr_chunks_chromadb',
#     embedding_function=HuggingFaceEmbeddings(model_name="google/embeddinggemma-300m")
#     # GoogleGenerativeAIEmbeddings(
#     #     model="models/gemini-embedding-001", 
#     #     google_api_key=os.getenv('GEMINI_API_KEY')
#     # )
# )
from langchain_elasticsearch import ElasticsearchStore
from langchain_elasticsearch import DenseVectorStrategy

vector_store = ElasticsearchStore(
    es_cloud_id=os.getenv('ELASTIC_CLOUD_ID'),
    es_api_key=os.getenv('ELASTIC_API_KEY'),
    index_name="langchain_index",
    embedding=embedding,
    strategy=DenseVectorStrategy(hybrid=True)
)

# TODO
def ocr_uppsala_brochure_analyzer():
    # pdf parsing from https://destinationuppsala.se/wp-content/uploads/2025/09/besokskarta-destination-uppsala-2025.pdf
    pass

def webscr_metadata_extractor(record, metadata):
    metadata["title"] = record.get("title") # These are also important fields that should be added
    metadata["url"] = record.get("url") # can help in website-based RAG later
    return metadata

docs = JSONLoader(
    file_path='./uppsala_chunks.json',
    jq_schema='.[]',
    content_key="text",
    metadata_func=webscr_metadata_extractor
).load()

non_duplicated_docs = []
contents = set()

for doc in docs:
    current = doc.page_content
    if current in contents:
        pass
    else:
        if len(current.split(" ")) > 3:
            contents.add(current)
            non_duplicated_docs.append(doc)

num_batches = 48
for i in range(1,num_batches): #
    start = int((i-1)*(len(non_duplicated_docs)//(num_batches-1)))
    end = int(i*(len(non_duplicated_docs)//(num_batches-1)))
    
    vector_store.add_documents(documents=non_duplicated_docs[start:end])
    print("Embedded from", start, ":", end, ">> on batch:", i, "of", num_batches-1)