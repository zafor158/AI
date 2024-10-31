# -*- coding: utf-8 -*-
import torch
from transformers import BitsAndBytesConfig, pipeline
import whisper
import gradio as gr
import numpy as np
import datetime
import re
from PIL import Image
from gtts import gTTS

# Set up quantization configuration
quantization_config = BitsAndBytesConfig(
    load_in_4bit=False,
    bnb_4bit_compute_dtype=torch.float16
)

model_id = "llava-hf/llava-1.5-7b-hf"
pipe = pipeline("image-to-text", model=model_id, model_kwargs={"quantization_config": quantization_config})

# Check device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using torch {torch.__version__} ({DEVICE})")

# Load Whisper model
model = whisper.load_model("medium", device=DEVICE)

# Logger file setup
tstamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
logfile = f'{tstamp}_log.txt'

def writehistory(text):
    with open(logfile, 'a', encoding='utf-8') as f:
        f.write(text + '\n')

def img2txt(input_text, input_image):
    # Load the image
    image = Image.open(input_image)
    
    writehistory(f"Input text: {input_text}")
    
    if isinstance(input_text, tuple):
        prompt_instructions = """
        Describe the image using as much detail as possible, is it a painting, a photograph, what colors are predominant, what is the image about?
        """
    else:
        prompt_instructions = f"""
        Act as an expert in imagery descriptive analysis, using as much detail as possible from the image, respond to the following prompt:
        {input_text}
        """
    
    writehistory(f"prompt_instructions: {prompt_instructions}")
    prompt = "USER: <image>\n" + prompt_instructions + "\nASSISTANT:"
    
    outputs = pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": 200})

    # Properly extract the response text
    if outputs and len(outputs[0]["generated_text"]) > 0:
        match = re.search(r'ASSISTANT:\s*(.*)', outputs[0]["generated_text"])
        if match:
            return match.group(1)
        else:
            return "No response found."
    else:
        return "No response generated."

def transcribe(audio):
    if audio is None or audio == '':
        return '', None  # Return empty strings for missing audio

    audio = whisper.load_audio(audio)
    audio = whisper.pad_or_trim(audio)
    mel = whisper.log_mel_spectrogram(audio).to(model.device)
    result = whisper.decode(model, mel, whisper.DecodingOptions())
    return result.text

def text_to_speech(text, file_path):
    audioobj = gTTS(text=text, lang='en', slow=False)
    audioobj.save(file_path)
    return file_path

def process_inputs(audio_path, image_path):
    speech_to_text_output = transcribe(audio_path)
    
    if image_path:
        chatgpt_output = img2txt(speech_to_text_output, image_path)
    else:
        chatgpt_output = "No image provided."
    
    processed_audio_path = text_to_speech(chatgpt_output, "Temp3.mp3")
    return speech_to_text_output, chatgpt_output, processed_audio_path

# Create the Gradio interface
iface = gr.Interface(
    fn=process_inputs,
    inputs=[gr.Audio(sources=["microphone"], type="filepath"),
            gr.Image(type="filepath")],
    outputs=[gr.Textbox(label="Speech to Text"),
             gr.Textbox(label="ChatGPT Output"),
             gr.Audio("Temp.mp3")],
    title="Learn OpenAI Whisper: Image processing with Whisper and Llava",
    description="Upload an image and interact via voice input and audio response."
)

# Launch the interface
iface.launch(debug=True)
