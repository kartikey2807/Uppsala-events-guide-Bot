## This code is to be run only once or possibly, minimal number of times (as embedding rate limits are finite). https://python.langchain.com/docs/integrations/document_loaders/json/#extracting-metadata
import os
from langchain_community.document_loaders import JSONLoader
from dotenv import load_dotenv
load_dotenv() # should make a .env file, and store GEMINI_API_KEY=AIe4..., ELASTIC_CLOUD_ID, ELASTIC_API_KEY.

from langchain_elasticsearch import ElasticsearchStore, DenseVectorStrategy
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
embedding = HuggingFaceEmbeddings(model_name="google/embeddinggemma-300m")

# Connect to Vector store
vector_store = ElasticsearchStore(
    es_cloud_id=os.getenv('ELASTIC_CLOUD_ID'),
    es_api_key=os.getenv('ELASTIC_API_KEY'),
    index_name="langchain_index",
    embedding=embedding,
    strategy=DenseVectorStrategy(hybrid=True) # improved vector retrieval (with occassional literal matching-retrieval also)
)

def webscr_metadata_extractor(record, metadata):
    metadata["title"] = record.get("title") # These are also important fields that should be added
    metadata["url"] = record.get("url") # can help in website-based RAG later
    return metadata

# To load the webscraped .json for splitting in documents
docs = JSONLoader(
    file_path='./uppsala_chunks.json',
    jq_schema='.[]',
    content_key="text",
    metadata_func=webscr_metadata_extractor
).load()

non_duplicated_docs = []
contents = set()

# Iterate through the JSON-split objects
for doc in docs:
    current = doc.page_content
    if current in contents: # duplicate documents detection loop
        pass
    else:
        # if not exists from earlier, and has a length of at least 4 words ...
        if len(current.split(" ")) > 3:
            contents.add(current)
            non_duplicated_docs.append(doc)

# embed + add documents in chunks for faster loading
num_batches = 48
for i in range(1,num_batches): #
    start = int((i-1)*(len(non_duplicated_docs)//(num_batches-1)))
    end = int(i*(len(non_duplicated_docs)//(num_batches-1)))
    
    # Push documents to the elastic vector store
    vector_store.add_documents(documents=non_duplicated_docs[start:end])
    print("Embedded from", start, ":", end, ">> on batch:", i, "of", num_batches-1)