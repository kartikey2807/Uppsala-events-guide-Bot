from __future__ import annotations
from typing import Iterable
import gradio as gr
from gradio.themes.base import Base
from gradio.themes.utils import colors, fonts, sizes
import time, base64, json
from typing import Iterator
from dotenv import load_dotenv
load_dotenv() # should make a .env file, and store GEMINI_API_KEY=AIe4...

#from google import genai
#from google.genai import types

import gradio as gr, os
from gradio import ChatMessage
from PIL import Image
import chromadb

from langchain_community.document_loaders import JSONLoader
from langchain_google_genai import ChatGoogleGenerativeAI  
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain.memory import ConversationBufferMemory
from langchain_community.document_loaders import JSONLoader
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_chroma import Chroma

client = chromadb.PersistentClient(path="./webscr_chunks_chromadb")
embedding = HuggingFaceEmbeddings(model_name="google/embeddinggemma-300m")
vector_store_from_client = Chroma(
    client=client,
    collection_name="langchain",
    embedding_function=embedding
)

def stream_gemini_response(user_message: str, messages: list, file) -> Iterator[list]:
    temp = 0.8
    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-flash",
        temperature=temp,
        timeout=None,
        max_retries=2,
        google_api_key=os.getenv('GEMINI_API_KEY'),
        model_kwargs={
            "max_output_tokens": 2048,  
            "top_k": 10,
            "top_p": 0.95
        })

    # TODO
    # chain_with_history = RunnableWithMessageHistory(
    #     llm,
    #     # Uses the get_by_session_id function defined in the example
    #     # above.
    #     get_by_session_id,
    #     input_messages_key="question",
    #     history_messages_key="history",
    # )

    # get similar documents
    # docs = vector_store_from_client.similarity_search_by_vector(
    #     embedding=embedding.embed_query(user_message), k=5
    # )
    retriever = vector_store_from_client.as_retriever(
        search_type="mmr", search_kwargs={"k": 5, "lambda_mult": 0.2}
    ) # retrieve potentially more docs
    docs = retriever.invoke(user_message)

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
        history.append(HumanMessage(
            content=f'{user_message if user_message != "" else ""}' + "\n<context>\n" + "".join(["- " + context.page_content + " \n" for context in docs]) + "\n</context>\n" + "\n\nHistory:\n" + "".join([f'{record.content}\n' for record in history]) # TODO: IMPLEMENT HISTORY AS LANGCHAIN MODULE
        ))

    print(history[-1].content)

    response = ""
    for chunk in llm.stream([history[-1]]):
        response += chunk.content
        yield response
    history.append(AIMessage(content=response))

########################################
# The theme of this Uppsala Info App
class Seafoam(Base):
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
            block_background_fill="rgba(255,255,255,0.4)",
            block_background_fill_dark="rgba(0,0,0,0.4)",
            block_padding="*spacing_xl calc(*spacing_xl)",
            body_background_fill="url('https://i.postimg.cc/2864Fs2p/image-1.jpg') rgba(255,255,255,0.6) no-repeat center / cover padding-box fixed",
            body_background_fill_dark="url('https://i.postimg.cc/JnMrcW3X/image.jpg') rgba(255,255,255,0.6) no-repeat center / cover padding-box fixed", # https://commons.wikimedia.org/wiki/File:Uppsala_domkyrka_December_2024_01.jpg
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
            body_text_color="#000000",
            body_text_color_dark="#FFFFFF",
            body_text_color_subdued="#4b5563",
            body_text_color_subdued_dark="#af5104",
            border_color_accent_subdued="#ffbeaa",
            color_accent_soft="#ffbeaa"
        )

seafoam = Seafoam()

# image credit: https://destinationuppsala.mediaflowportal.com/folder/147814/
# https://destinationuppsala.mediaflowportal.com/folder/281165/ Gustav Dalesjö

with gr.Blocks(theme=seafoam, fill_height=True) as demo:
    history = [
        SystemMessage(content="""You are an assistant bot that is to discuss about the city of Uppsala in Sweden. Your objective is to give accurate and up-to-date information to the user, who is aiming to get to know to attend events or explore places in Uppsala, and have a wonderful time in their visit. You are providing information and/or data only about the city of Uppsala, and should not endorse other cities in the country of Sweden. Use the context below also to help you answer. If the context doesn't contain any relevant information to the question about Uppsala, don't make something up and just say that you do not know.""")
    ]

    image = gr.Image(
        type="filepath", 
        height=200, 
        sources=["upload", "clipboard"], 
        #visible='hidden'
    )
    question = gr.Textbox(placeholder="Type your message here and press Enter...")

    with gr.Column(scale=5):    
        with gr.Row(height="300px"):
            chatbot = gr.Chatbot(type="messages", autoscroll=True)
            chat_interface = gr.ChatInterface(
                stream_gemini_response,
                type="messages",
                title="Uppsala Event Guide: The official guide to explore Uppsala",
                multimodal=True,
                chatbot=chatbot,
                textbox=question,
                additional_inputs=[
                    image
                ],
                theme=seafoam,
                fill_height=True
            )
        with gr.Row(height="300px"):
            with gr.Column():
                question
            with gr.Column():
                image
#########################################

if __name__ == "__main__":
    demo.launch(debug=True)
