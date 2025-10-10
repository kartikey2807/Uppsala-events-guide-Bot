from __future__ import annotations
from typing import Iterable
import gradio as gr
from gradio.themes.base import Base
from gradio.themes.utils import colors, fonts, sizes
import time

from typing import Iterator
from dotenv import load_dotenv
load_dotenv() # should make a .env file, and store GEMINI_API_KEY=AIe4...

from google import genai
from google.genai import types

import gradio as gr, os
from gradio import ChatMessage
from PIL import Image

# Adapted from 
# https://www.gradio.app/guides/agents-and-tool-usage#a-real-example-using-gemini-2-0-flash-thinking-api
def stream_gemini_response(user_message: str, messages: list, file) -> Iterator[list]:
    temp = 0.8
    history.append(ChatMessage(role="user", content=user_message))
    
    instructions = "You are an assistant bot that is to discuss about the city of Uppsala in Sweden. Your objective is to give accurate and up-to-date information to the user, who is aiming to get to know to attend events or explore places in Uppsala, and have a wonderful time in their visit. You are providing information and/or data only about the city of Uppsala, and should not endorse other cities in the country of Sweden."
    
    prompt = f'{"Question: " + history[-1].content if history[-1].content != "" else ""}' + "\n\nHistory:\n" + "".join([record.content for record in history[:-1]])

    # Initially set the payload to be the prompt
    payload = prompt

    if file:
        print("Image was uploaded. Continuing....")
        # If an image was uploaded, update the value of payload to be now an 
        # array of [prompt, file (in PIL Image form)] 
        payload = [prompt, Image.open(file)]

    client = genai.Client()
    model = "gemini-2.0-flash"
    config = types.GenerateContentConfig(
                system_instruction=instructions,
                temperature=temp,
                # Safety feedback -- generateContent returns a GenerateContentResponse which includes safety feedback.
                safety_settings=[
                    types.SafetySetting(
                       category='HARM_CATEGORY_DANGEROUS_CONTENT',
                       threshold='BLOCK_ONLY_HIGH', # 'BLOCK_LOW_AND_ABOVE'
                    )
                ]
            )
 
    # Initialize buffers
    thought_buffer = ""
    response_buffer = ""
    thinking_complete = False
    
    # Add initial thinking message
    history.append(
        ChatMessage(
            role="assistant",
            content=""
        )
    )

    for chunk in client.models.generate_content_stream(model=model, contents=payload, config=config):
        try:
            parts = chunk.candidates[0].content.parts
            current_chunk = parts[0].text

            if len(parts) == 2 and not thinking_complete:
                # Complete thought and start response
                thought_buffer += current_chunk
                history[-1] = ChatMessage(
                    role="assistant",
                    content=thought_buffer
                )
                
                # Add response message
                history.append(
                    ChatMessage(
                        role="assistant",
                        content=parts[1].text
                    )
                )
                thinking_complete = True
                
            elif thinking_complete:
                # Continue streaming response
                response_buffer += current_chunk
                history[-1] = ChatMessage(
                    role="assistant",
                    content=response_buffer
                )
                
            else:
                # Continue streaming thoughts
                thought_buffer += current_chunk
                history[-1] = ChatMessage(
                    role="assistant",
                    content=thought_buffer
                )

            yield history[-1]
        except Exception:
            if chunk.usage_metadata:
                print("\n>>>>>>>>>>>>>>>>>>> Safety Filter Triggered <<<<<<<<<<<<<<<<<<<<<<<")
                print(chunk) # Print what safety filters were violated in just the terminal
                print(">>>>>>>>>>>>>>>>>>> *********************** <<<<<<<<<<<<<<<<<<<<<<<")
            yield "There was an error. Please try again."
            

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
            fonts.GoogleFont("IBM Plex Mono"),
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
            body_background_fill="url('https://i.postimg.cc/tJwwqcpf/Screenshot-2025-10-09-at-10-30-28.png') rgba(255,255,255,0.6) no-repeat center / cover padding-box fixed",
            body_background_fill_dark="url('https://i.postimg.cc/bvM1hGZ4/Screenshot-2025-10-10-at-14-18-52.png') rgba(255,255,255,0.6) no-repeat center / cover padding-box fixed",
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
            body_text_color="#261F04",
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
    history = []

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
