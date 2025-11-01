## Visit Uppsala - Guide Chatbot &nbsp; [![Open in Spaces](https://huggingface.co/datasets/huggingface/badges/resolve/main/open-in-hf-spaces-md.svg)](https://huggingface.co/spaces/adityak714/visit-uppsala)

Group 4 - 1RT730 LLMs and Societal Consequences of AI - Project 

<img width="1522" height="877" alt="Screenshot 2025-10-19 at 20 05 44" src="https://github.com/user-attachments/assets/810968ae-9737-4e71-8e84-a6840435242d" />
<img width="1526" height="667" alt="Screenshot 2025-10-19 at 04 55 03" src="https://github.com/user-attachments/assets/d41223fd-d48c-4462-b3e1-afdad1f920e3" />

<hr>
The current project develops a LLM application that allows users to ask questions for information or guidance about the city of Uppsala, Sweden. Finding general information, specific activities and things to do in Uppsala can be tedious and difficult, especially for non-swedish speaking people. You would have to find and navigate through websites like Destination Uppsala and Uppsala Kommun. These sites are dense with information but are cumbersome to traverse. Nowadays, more and more individuals use Generative AI-based assistants for general information, tourism-related R&D and itinerary planning, among several other day-to-day tasks. 

<hr>
Two instances of Language Models are part of the chat interface; gemini-2.5-flash for user-side generation, and a secondary gemini-2.5-flash client handling Google-Search based-retrieval. The chatbot’s User Interface was made using the Gradio library (Python), and the AI Application Framework used for this project is Langchain (Python). The choice for the vector DB is Elastic DB, with a cloud-hosted instance (deployed on Google Cloud), storing all documents from webscraped data and the OCR-processed information PDF, with the text attribute, vector, as well as metadata (url, and title). For the embedding model, `google/embeddinggemma-300m` from Huggingface was cloned. To improve document retrieval (as certain information was in Swedish, while some in English), a translation API tool detect-language was used in the `chatbot.py` file applied on a query (more efficient approach to match documents rather than running a translation conversion process on all documents, which would not be feasible for very large data scenarios).

<hr>
To get started, simply make a .env file in the `dest-uppsala-crawler`, or the root folder (should be same as where the .py files are located). 

It needs to include:

- ELASTIC_CLOUD_ID, ELASTIC_API_KEY (to connect to the vector database ElasticDB hosted on-cloud)
- GEMINI_API_KEY, GEMINI_API_KEY_2 (for the two LLM instances powering this application)
- TRANSLATOR_KEY (for the translation API used in RAG for getting more relevant documents that may be left out due to language difference)

Next, download the necessary packages as per the requirements on Python. You should have a virtual environment or similar up and prepared. To run the chatbot, run the following code:

```
(venv) $ gradio chatbot.py
```

Or

```
(venv) $ python chatbot.py
```

The response should be of a Gradio interface launched at http://127.0.0.1:7860/ or similar. Open it on a suitable web browser. Once you see the Chat Interface, go on away and ask anything to know about the city of Uppsala!

**For developers:** For the code files done for data processing approaches, RAG, benchmark testing, evaluation and document creation / chunking / embedding, view the relevant files in Python, which have commented lines that explain what is being done in each stage. 

### Architecture:

<img width="864" height="738" alt="Screenshot 2025-10-19 at 21 33 02" src="https://github.com/user-attachments/assets/cedbbb26-f6bb-46fb-b580-8e275e2cbdbe" />

### Contributors:
- Aditya Khadkikar
- Alexander Sundquist
- Kartikey Sharma
