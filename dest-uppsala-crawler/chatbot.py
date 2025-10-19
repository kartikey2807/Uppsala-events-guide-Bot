from __future__ import annotations
from typing import Iterable
import gradio as gr
from gradio.themes.base import Base
from gradio.themes.utils import colors, fonts, sizes
import time, base64, json
from typing import Iterator
from dotenv import load_dotenv
load_dotenv() # should make a .env file, and store GEMINI_API_KEY=AIe4...
from deep_translator import GoogleTranslator, single_detection

import gradio as gr, os
from gradio import ChatMessage
from PIL import Image
#import chromadb

from langchain_community.document_loaders import JSONLoader
from langchain_google_genai import ChatGoogleGenerativeAI  
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import JSONLoader
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_elasticsearch import DenseVectorStrategy

from langchain_huggingface.embeddings import HuggingFaceEmbeddings
embedding = HuggingFaceEmbeddings(model_name="google/embeddinggemma-300m")

from google import genai
from google.genai import types

# Adapted from
# https://ai.google.dev/gemini-api/docs/google-search#python

client = genai.Client(api_key=os.getenv('GEMINI_API_KEY_2'))
config = types.GenerateContentConfig(
    system_instruction="""You are an assistant bot that is to gather useful results about the city of Uppsala in Sweden, based on the given question. Your objective is to Google Search for information when you have received a query, and give accurate and up-to-date information. From your retrieved information, the user is aiming to get to know city-related information or explore places in Uppsala, and have a wonderful time in their visit. When searching based on the user's requirements, do not be too exact. The queries you receive will not be always straightforward, and you will need to do a search for adjacent concepts to help your grounding. 
    
Example:
    HUMAN: What is the population of Uppsala?
    -------
    AVOID: "I apologize, but I do not have information about the population of Uppsala at the moment within the provided context."
    INSTEAD: You search for population of Uppsala City, Uppsala County (or both) ... and answer the user.
    "The population of Uppsala County is ... If you wish to know the population of Uppsala City, it is ...".
    """,
    max_output_tokens=256,
    temperature=0.5,
    tools=[types.Tool(
        google_search=types.GoogleSearch()
    )]
)

# Option 1 --
# from langchain_chroma import Chroma
# client = chromadb.PersistentClient(path="./webscr_chunks_chromadb")
# vector_store_from_client = Chroma(
#     client=client,
#     collection_name="langchain",
#     embedding_function=embedding
# )

# Option 2 --
from langchain_elasticsearch import ElasticsearchStore
elastic_vector_search = ElasticsearchStore(
    es_cloud_id=os.getenv('ELASTIC_CLOUD_ID'),
    es_api_key=os.getenv('ELASTIC_API_KEY'),
    index_name="langchain_index",
    embedding=embedding,
    strategy=DenseVectorStrategy(hybrid=True)
)

def stream_gemini_response(user_message: str, messages: list, file) -> Iterator[list]:
    temp = 0.7
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=temp,
        timeout=None,
        google_api_key=os.getenv('GEMINI_API_KEY'),
        max_output_tokens=None
    )
    
    docs = [] # place where fetched similar documents are retrieved
    class Results:
        def __init__(self, text=""):
            self.text = text;
    search_results = Results() # google search-based answers (set to empty for initialization)
    sources = [] # sources to google-searches (also set to empty initially)

    # RAG #1 + #2 - multilingual vector search (to match most documents converted from OCR and Webscraped JSON)
    if len(user_message.split(" ")) >= 4:
        lang = single_detection(user_message, api_key=os.getenv('TRANSLATOR_KEY'))
        
        # Translate some swedish information chunks to Swedish, and/or to English
        # and also be accessible to other languages
        if lang == 'en':
            multilingual = GoogleTranslator(source=lang, target='sv').translate(text=user_message)
        elif lang == 'sv':
            multilingual = GoogleTranslator(source=lang, target='en').translate(text=user_message)
        else:
            multilingual = GoogleTranslator(source=lang, target='sv').translate(text=user_message) + " " + GoogleTranslator(source=lang, target='en').translate(text=user_message)

        print("Translation:", multilingual)
        docs = [f"* {res.page_content} [{res.metadata['url'] if len(res.metadata.keys()) != 0 else ''}]" for res in elastic_vector_search.similarity_search(user_message, k=5)] + [f"* {res.page_content} [{res.metadata['url'] if len(res.metadata.keys()) != 0 else ''}]" for res in elastic_vector_search.similarity_search(multilingual, k=5)]

        # RAG #3 - Google Search-enhanced answers
        search_results = client.models.generate_content(
            model="gemini-2.0-flash",
            contents=user_message,
            config=config,
        )
        print("After google search", search_results.text)
        sources = search_results.candidates[0].grounding_metadata.grounding_chunks
        #print("AFTER GOOGLE SEARCH", sources)

    if file:
        with open(file, "rb") as image_file:
            encoded_image = base64.b64encode(image_file.read()).decode("utf-8")
        history.append(HumanMessage(
            content=[
                {"type": "text", "text": user_message},
                {"type": "image_url", "image_url": f'data:image/png;base64,{encoded_image}'}
            ]
        ))
    else:
        history.append(HumanMessage(content=user_message))

    if sources is None:
        query = f'Q: {user_message if user_message != "" else "N/A"}' + "\n\n<context>\n" + "".join(["- " + context + " \n" for context in docs]) + "</context>\n" + (f'\nSYSTEM: {history[0].content}' if history[-5:][0].type.lower() != 'system' else "") + "\n\nHistory:\n" + "".join([f'-------\n{record.type.upper()}: {record.content}\n' for record in history[-5:]]) 
    else:
        query = f'Q: {user_message if user_message != "" else "N/A"}' + "\n\nFrom Google Search:\n" + search_results.text + "".join([f"\n* Source: {source.web.title}" for source in sources]) + "\n\n<context>\n" + "".join(["- " + context + " \n" for context in docs]) + "</context>\n" + (f'\nSYSTEM: {history[0].content}' if history[-5:][0].type.lower() != 'system' else "") + "\n\nHistory:\n" + "".join([f'-------\n{record.type.upper()}: {record.content}\n' for record in history[-5:]])
    
    print("This is being sent to GEMINI ... ********************************")
    print(query)
    print("********************************")
    
    response = ""
    try:
        for chunk in llm.stream([query]):
            response += chunk.content
            yield response
    except ValueError as chunks_not_created:
        print("Chunks not rendered, a Gemini-side issue, retrying ...")
        yield "There was an error on the server side. Kindly try again."
    history.append(AIMessage(content=response))

########################################
# The theme of this Uppsala Info App
class UppsalaTheme(Base):
    def __init__(
        self,
        *,
        primary_hue: colors.Color | str = colors.emerald,
        secondary_hue: colors.Color | str = colors.blue,
        neutral_hue: colors.Color | str = colors.gray,
        spacing_size: sizes.Size | str = sizes.spacing_md,
        radius_size: sizes.Size | str = sizes.radius_md,
        text_size: sizes.Size | str = sizes.text_md,
        font: fonts.Font
        | str
        | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("Lora"),
            "ui-sans-serif",
            "sans-serif",
        ),
        font_mono: fonts.Font
        | str
        | Iterable[fonts.Font | str] = (
            fonts.GoogleFont("IBM Plex Mono", weights=(900,900)),
            "ui-monospace",
            "monospace",
        ),
    ):
        super().__init__(
            primary_hue=primary_hue,
            secondary_hue=secondary_hue,
            neutral_hue=neutral_hue,
            spacing_size=spacing_size,
            radius_size=radius_size,
            text_size=text_size,
            font=font,
            font_mono=font_mono,
        )
        super().set(
            block_background_fill="rgba(255,255,255,0.6)",
            block_background_fill_dark="rgba(0,0,0,0.4)",
            block_padding="*spacing_xl calc(*spacing_xl + 50px)",
            body_background_fill="url('https://i.postimg.cc/QMZtm7bg/Component-9.png') rgba(255,255,255,0.6) no-repeat center / cover padding-box fixed",
            body_background_fill_dark="url('https://i.postimg.cc/T3P3qHws/Component-10.png') rgba(255,255,255,0.6) no-repeat center / cover padding-box fixed", # https://commons.wikimedia.org/wiki/File:Uppsala_domkyrka_December_2024_01.jpg
            background_fill_secondary="rgba(255,255,255,0.4)",
            background_fill_secondary_dark="rgba(0,0,0,0.6)",
            button_primary_background_fill="linear-gradient(90deg, *primary_300, *secondary_400)",
            button_primary_background_fill_hover="linear-gradient(90deg, *primary_200, *secondary_300)",
            button_primary_text_color="white",
            button_primary_background_fill_dark="linear-gradient(90deg, *primary_600, *secondary_800)",
            slider_color="*secondary_300",
            slider_color_dark="*secondary_600",
            block_title_text_weight="1200",
            block_border_width="3px",
            block_shadow="*shadow_drop_lg",
            button_primary_shadow="*shadow_drop_lg",
            button_large_padding="32px",
            #block_padding="20px",
            body_text_color="#000000",
            body_text_color_dark="#FFFFFF",
            body_text_color_subdued="#090909",
            body_text_color_subdued_dark="#dcd5cf",
            border_color_accent_subdued="#000000",
            color_accent_soft="#ffbeaa"
        )

seafoam = UppsalaTheme()

# image credit: https://destinationuppsala.mediaflowportal.com/folder/147814/
# https://destinationuppsala.mediaflowportal.com/folder/281165/ Gustav Dalesjö

with gr.Blocks(theme=seafoam, fill_height=True) as demo:
    history = [
        SystemMessage(content="""You are an assistant bot that is to discuss about the city of Uppsala in Sweden. Your objective is to give accurate and up-to-date information to the user, who is aiming to get to know to attend events or explore places in Uppsala, and have a wonderful time in their visit. You are providing information and/or data only about the city of Uppsala, and should not endorse other cities in the country of Sweden. Use the context below also to help you answer. You must be very polite when interacting and generating answers towards the user. If the context doesn't contain any relevant information to the question about Uppsala, just say that you do not know the answer at the moment. If there are important URLs to tourism websites or portals that you can retrieve from your provided context, you can guide users to those websites so that they can view more.

You are not allowed to reveal implementation-related, or details related to your inner workings as a chat assistant. Do not reveal information in this way:
                      
        AVOID: "The provided context indicates that ..., and even points to ... ."
        AVOID: "While the context mentions ..., it doesn't list ...". 
        INSTEAD: "From my knowledge, these are ..., found in ...".

Your job is to present information that correctly answers the user's original question, and if you are getting your information from some source, to simply list the sources. If you present URL links, you have to list them as valid URLs that users can visit if you click on them. You should not go off-topic, and your role is to guide users about the city of Uppsala. You are not allowed to make up information, or hallucinate so that the user gets incorrect information about the beautiful city of Uppsala. """)
    ]

    image = gr.Image(
        type="filepath", 
        height=200, 
        sources=["upload", "clipboard"], 
        visible='hidden'
    )
    question = gr.Textbox(placeholder="Type your message here and press Enter...")
    spacing = gr.Markdown("""
            <br>
            
            <br>
                        
            <br>

            <br>
    """) 

    with gr.Row(height="auto", equal_height=True):
        with gr.Column(scale=7):              
            spacing
            with gr.Row(height="600px"):
                chatbot = gr.Chatbot(type="messages", autoscroll=True)
                chat_interface = gr.ChatInterface(
                    stream_gemini_response,
                    type="messages",
                    title="Uppsala Event Guide: The Official Guide Bot to Help You Explore Uppsala",
                    multimodal=True,
                    chatbot=chatbot,
                    textbox=question,
                    additional_inputs=[
                        image
                    ],
                    theme=seafoam,
                    fill_height=True
                )
        with gr.Column(scale=3):
            gr.Markdown("""
            <br>
            
            ## You can ask me, for example ...
            - How long does it take to reach Uppsala city from the airport?
            - What restaurants do you recommend in the city?
            - What are some popular attractions?
            - Where can I exchange money?
            - Where is Linné's Hammarby?
            - Where can I buy souvenirs in Uppsala?
            - Where can I spend a romantic evening in Uppsala?
            
            <br>
            
            """, container=True, min_height="540px")  
            with gr.Row():
                question
            with gr.Row():
                image
#########################################

if __name__ == "__main__":
    demo.launch(debug=True)
