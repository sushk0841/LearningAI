import streamlit as st
import torch
import librosa
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import numpy as np
import pandas as pd
from tabulate import tabulate
import json

# Check for GPU availability
device = "cuda:0" if torch.cuda.is_available() else "cpu"
torch_dtype = torch.float16 if torch.cuda.is_available() else torch.float32

# Model initialization
model_id = "openai/whisper-large-v3"
model = AutoModelForSpeechSeq2Seq.from_pretrained(
    model_id, torch_dtype=torch_dtype, low_cpu_mem_usage=True, use_safetensors=True
)
model.to(device)

# Processor initialization
processor = AutoProcessor.from_pretrained(model_id)

# Pipeline initialization
# Pipeline initialization
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps="word",  # Change this line to return word-level timestamps
    torch_dtype=torch_dtype,
    device=device,
)

# File upload
uploaded_file = st.file_uploader("Upload an audio file", type="wav")

if uploaded_file is not None:
    # Load the audio file using librosa
    input_audio, _ = librosa.load(uploaded_file, sr=16000)
    
    st.audio(uploaded_file)

    # Perform speech recognition with timestamps
    result = pipe(input_audio)

    # Recognized text
    recognized_text = result["text"]
    print("Recognized text:", recognized_text)
    print("\n")

    # Extract timestamps
    timestamps = result.get("chunks", [])
    print("Timestamps:", timestamps)

    # Chunks with timestamps
    chunks = result.get("chunks", [])
    print("\n")
    
    # Red flag categories
    red_flags = [
        
    {
        "Phrase": "Totally unqualified",
        "Category": "professionalism"
    },
    {
        "Phrase": "fucking",
        "Category": "frustration"
    }
    ]

    # Red flag detection
    red_flag_matches = []
    for i, chunk in enumerate(chunks):
        chunk_text = chunk['text']
        for red_flag in red_flags:
            if red_flag['Phrase'] in chunk_text:
                red_flag_matches.append([red_flag['Phrase'], red_flag['Category'], timestamps[i]])

    # Output
    output_text = "Red flags detected: " + "\n".join([f"{match[0]} (professionalism) at {match[2]}" for match in red_flag_matches]) if red_flag_matches else "No red flags detected."
    df_red_flag_matches = pd.DataFrame(red_flag_matches, columns=['Phrase', 'Category', 'Timestamp'])
    df_red_flag_matches['Timestamp'] = df_red_flag_matches['Timestamp'].apply(lambda x: x[0] if isinstance(x, list) else None)

    # Display the recognized text, timestamps, and red flag matches
    st.write("Recognized Text:")
    st.write(recognized_text)
    st.write("Timestamps:")
    st.write(timestamps)
    st.write("Red Flag Matches:")
    st.write(df_red_flag_matches)