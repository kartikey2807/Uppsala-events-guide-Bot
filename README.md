## Visit Uppsala - Guide Chatbot
Group 4 - 1RT730 LLMs and Societal Consequences of AI Project

The current project develops a LLM application that allows users to ask questions for information or guidance about the city of Uppsala, Sweden. Finding general information, specific activities and things to do in Uppsala can be tedious and difficult, especially for non-swedish speaking people. You would have to find and navigate through websites like Destination Uppsala and Uppsala Kommun. These sites are dense with information but are cumbersome to traverse. 

To get started, simply make a .env file in the `dest-uppsala-crawler`, or the root folder (should be same as where the .py files are located). It needs to include:

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

Contributors:
- Aditya Khadkikar
- Alexander Sundquist
- Kartikey Sharma