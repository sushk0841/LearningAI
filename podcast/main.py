import torch
import librosa
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import numpy as np
from tabulate import tabulate
import csv

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
pipe = pipeline(
    "automatic-speech-recognition",
    model=model,
    tokenizer=processor.tokenizer,
    feature_extractor=processor.feature_extractor,
    max_new_tokens=128,
    chunk_length_s=30,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)

# Load the audio file using librosa
input_audio, _ = librosa.load('sound\chunk_5.wav', sr=16000)

# Perform speech recognition with timestamps
result = pipe(input_audio)

# Recognized text
recognized_text = result['text']

# Read recognized text from file
with open('podcast/Transcripts/transcript.txt', 'w') as f:
    f.write(recognized_text)


print("Recognized text has been saved to 'transcript.txt'\n")

print("Below is the Timestamps of the audio file transcription:\n")

# Extract timestamps
timestamps = result.get('chunks', [])
print("Timestamps:", timestamps)

# Chunks with timestamps
chunks = result.get('chunks', [])
print("\n")


# Red flag data (you can add your red flag data here)
red_flag_data =[]

# Load red flag data from CSV file
with open('podcast\\CSV_DATA\\red_flags_data.csv', newline='', encoding='utf-8') as csvfile:
    reader = csv.reader(csvfile)
    for row in reader:
        if len(row) == 2:
            red_flag_data.append([row[0], row[1]])

# Split recognized text into sentences
recognized_sentences = recognized_text.split()

avgdl = sum(len(sentence) for sentence in recognized_sentences) / len(recognized_sentences)
N = len(recognized_sentences)

def bm25(word, sentence, k=1.2, b=0.75):
    freq = sentence.count(word)
    tf = (freq * (k + 1)) / (freq + k * (1 - b + b * (len(sentence) / avgdl)))
    N_q = sum([doc.count(word) for doc in recognized_sentences])
    idf = np.log(((N - N_q + 0.5) / (N_q + 0.5)) + 1)
    return tf * idf

# Compute similarity scores
similarity_scores_with_categories = {}
for word, category in red_flag_data:
    scores = []
    for i, sentence in enumerate(recognized_sentences):
        score = bm25(word, sentence)
        if score > 0.0:  # Consider only scores above 0.0
            scores.append((i + 1, score))
    if scores:
        similarity_scores_with_categories[word] = (category, scores)


# Initialize a list to store table rows
table_data = []

# Counter for serial number
serial_number = 1

# Populate table data
for word, (category, scores) in similarity_scores_with_categories.items():
    for doc_id, score in scores:
        table_data.append([serial_number, score, word, category])
        serial_number += 1
      
# Load the hate speech detection model
hate_speech_detector = pipeline(
    "text-classification",
    model="badmatr11x/distilroberta-base-offensive-hateful-speech-text-multiclassification",
    tokenizer="badmatr11x/distilroberta-base-offensive-hateful-speech-text-multiclassification"
)

# Input text
input_text = recognized_text

# Perform hate speech detection on recognized text
hate_speech_result = hate_speech_detector(input_text)

# Print the table
print(tabulate(table_data, headers=["Serial No.", "Similarity Score", "Red Flags", "Category",""]))
print("\n")

# Print the result
print("Hate Speech Detection Result:", hate_speech_result)

