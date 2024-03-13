# Recognized text:  If there was one piece of advice you could give to EXP agents, what would that be? Entering your day with a lot of gratitude. I think that's the way I try to approach every day in life is to wake up grateful. You know, no matter where your market is, there's going to be someone that's going to need your help. there's going to be someone that's going to need your help. And, you know, you have to seek those people out. And you have to be able to show them a level of grace as well. Not everyone understands real estate. You know, we do it every day. So sometimes we just get into the motion, we just start talking our lingo, we start doing our thing and then the client is just confused. So I find that we have to kind of enter the day with that certain level of gratitude and talk to our clients with a little bit of grace. And I think that goes a long way.


# Below is the Timestamps of the audio file transcription:

# Timestamps: [{'timestamp': (0.0, 5.76), 'text': ' If there was one piece of advice you could give to EXP agents, what would that be?'}, {'timestamp': (6.76, 9.34), 'text': ' Entering your day with a lot of gratitude.'}, {'timestamp': (10.18, 17.22), 'text': " I think that's the way I try to approach every day in life is to wake up grateful."}, {'timestamp': (18.16, 21.92), 'text': " You know, no matter where your market is, there's going to be someone that's going to"}, {'timestamp': (21.92, 22.72), 'text': ' need your help.'}, {'timestamp': (25.34, 34.04), 'text': " there's going to be someone that's going to need your help. And, you know, you have to seek those people out. And you have to be able to show them a level of grace as well. Not everyone understands"}, {'timestamp': (34.04, 40.56), 'text': ' real estate. You know, we do it every day. So sometimes we just get into the motion, we just'}, {'timestamp': (40.56, 45.32), 'text': ' start talking our lingo, we start doing our thing and then the client is just'}, {'timestamp': (45.32, 47.22), 'text': ' confused.'}, {'timestamp': (47.22, 54.7), 'text': ' So I find that we have to kind of enter the day with that certain level of gratitude and'}, {'timestamp': (54.7, 57.52), 'text': ' talk to our clients with a little bit of grace.'}, {'timestamp': (57.52, 60.76), 'text': ' And I think that goes a long way.'}]


#   Serial No.    Similarity Score  Red Flags    Category
# ------------  ------------------  -----------  --------------------
#            1             2.61967  gratitude    telemarketing
#            2             2.79734  gratitude    telemarketing
#            3             3.93472  lingo        conflict of interest


# Hate Speech Detection Result: [{'label': 'NEITHER', 'score': 0.9999247789382935}]


import torch
import librosa
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
import numpy as np
from tabulate import tabulate
from datasets import load_dataset

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
    chunk_length_s=10,
    batch_size=16,
    return_timestamps=True,
    torch_dtype=torch_dtype,
    device=device,
)

# Text Summarization start from here...
summarizer = pipeline("summarization")

ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
sample = ds[0]["audio"]

# Load the audio file using librosa
input_audio, _ = librosa.load("sound/ramsay.wav", sr=16000)

# Perform speech recognition with timestamps
result = pipe(input_audio)

# Recognized text
recognized_text = result["text"]
print("Recognized text:", recognized_text)
print("\n")

print("Below is the Timestamps of the audio file transcription:\n")

# Extract timestamps
result = pipe(input_audio, return_timestamps="word")
print(result["chunks"])

print("Below is the Summarization of the audio file transcription:\n")
print("\n")

# defining function to summarize the transcription
def summarize_text(text):
    summary = summarizer(text, max_length=130, min_length=30, do_sample=False)
    return summary[0]['summary_text']

# Summarize the recognized text
summary = summarize_text(recognized_text)

print("Summary:\n" + summary)

print("\n")
print("\n")


# Red flag data (you can add your red flag data here)
red_flag_data = [
    # Professionalism and Conduct
    ["Totally unqualified", "professionalism"],
    ["She's a fraud", "professionalism"],
    ["They're useless", "professionalism"],
    ["Completely inept", "professionalism"],
    ["Total sham", "professionalism"],
    ["They're unethical", "professionalism"],
    ["What a disaster", "professionalism"],
    ["Unbelievably bad", "professionalism"],
    ["Totally shady", "professionalism"],
    ["They're a disgrace", "professionalism"],
    ["Absolute incompetence", "professionalism"],
    ["He's a cheat", "professionalism"],
    ["Such a scam", "professionalism"],
    ["Lacks integrity", "professionalism"],
    ["Totally unreliable", "professionalism"],
    ["Complete joke", "professionalism"],
    ["Unethical behavior", "professionalism"],
    ["Serious misconduct", "professionalism"],
    ["He's a charlatan", "professionalism"],
    ["They're corrupt", "professionalism"],
    ["Absolutely dishonest", "professionalism"],
    ["Lack of professionalism", "professionalism"],
    ["He's a swindler", "professionalism"],
    ["Totally unscrupulous", "professionalism"],
    ["She's deceitful", "professionalism"],
    ["They're fraudulent", "professionalism"],
    ["Completely unethical", "professionalism"],
    ["Utterly dishonest", "professionalism"],
    ["Blatantly corrupt", "professionalism"],
    ["Grossly incompetent", "professionalism"],
    # Income Claims and Financial Representations
    ["Instant wealth creation", "financial"],
    ["Surefire investment success", "financial"],
    ["Quick financial gain", "financial"],
    ["Rapid wealth accumulation", "financial"],
    ["Easy proﬁt-making", "financial"],
    ["Guaranteed return on investment", "financial"],
    ["Effortless financial growth", "financial"],
    ["Speedy income generation", "financial"],
    ["Rapid revenue increase", "financial"],
    ["Quick financial turnaround", "financial"],
    ["Instant money-making", "financial"],
    ["Guaranteed financial results", "financial"],
    ["Easy wealth accumulation", "financial"],
    ["Rapid proﬁt-making", "financial"],
    ["Effortless income boost", "financial"],
    ["Speedy proﬁt increase", "financial"],
    ["Quick revenue growth", "financial"],
    ["Instant financial improvement", "financial"],
    ["Guaranteed wealth boost", "financial"],
    ["Easy financial success", "financial"],
    ["Rapid wealth gain", "financial"],
    ["Quick proﬁt generation", "financial"],
    ["Instant revenue increase", "financial"],
    ["Guaranteed income boost", "financial"],
    ["Effortless wealth creation", "financial"],
    ["Speedy financial gains", "financial"],
    ["Quick wealth improvement", "financial"],
    ["Instant proﬁt increase", "financial"],
    ["Guaranteed revenue boost", "financial"],
    # Real Estate Telemarketing
    ["Blasting calls out", "telemarketing"],
    ["Spamming their inbox", "telemarketing"],
    ["Bombarding with texts", "telemarketing"],
    ["Nonstop robocalls", "telemarketing"],
    ["Endless phone spam", "telemarketing"],
    ["Persistent marketing tactics", "telemarketing"],
    ["gratitude", "telemarketing"],
    # RESPA Compliance
    ["Secret proﬁt sharing", "RESPA compliance"],
    ["Covert referral fee", "RESPA compliance"],
    ["Hidden kickback deal", "RESPA compliance"],
    ["Undercovered commission", "RESPA compliance"],
    ["Backhanded payment", "RESPA compliance"],
    ["Disguised referral fee", "RESPA compliance"],
    ["Cloaked compensation", "RESPA compliance"],
    ["Sneaky proﬁt split", "RESPA compliance"],
    ["Concealed kickback", "RESPA compliance"],
    ["Underhanded commission", "RESPA compliance"],
    ["Shrouded payment", "RESPA compliance"],
    ["Masked referral agreement", "RESPA compliance"],
    ["Veiled kickback arrangement", "RESPA compliance"],
    ["Hidden ﬁnancial incentive", "RESPA compliance"],
    ["Covert proﬁt exchange", "RESPA compliance"],
    ["Undercover referral agreement", "RESPA compliance"],
    ["Sly commission deal", "RESPA compliance"],
    ["Disguised proﬁt sharing", "RESPA compliance"],
    ["Cloaked ﬁnancial reward", "RESPA compliance"],
    ["Sneaky commission split", "RESPA compliance"],
    ["Concealed referral incentive", "RESPA compliance"],
    ["Underhanded proﬁt arrangement", "RESPA compliance"],
    ["Shrouded commission incentive", "RESPA compliance"],
    ["Masked kickback deal", "RESPA compliance"],
    ["Veiled ﬁnancial agreement", "RESPA compliance"],
    ["Hidden commission scheme", "RESPA compliance"],
    ["Covert referral incentive", "RESPA compliance"],
    ["Undercover proﬁt sharing", "RESPA compliance"],
    ["Sly kickback arrangement", "RESPA compliance"],
    ["Disguised commission incentive", "RESPA compliance"],
    # Fair Housing Laws
    ["Avoid renting to them", "fair housing laws"],
    ["Prefer not to sell to those people", "fair housing laws"],
    ["Don't deal with that community", "fair housing laws"],
    ["Exclude certain types", "fair housing laws"],
    ["Only rent to our kind", "fair housing laws"],
    ["Steer clear of that neighborhood", "fair housing laws"],
    ["Not suitable for certain families", "fair housing laws"],
    ["Don't want those kinds of tenants", "fair housing laws"],
    ["Avoid selling to that group", "fair housing laws"],
    ["Prefer not to work with them", "fair housing laws"],
    ["Exclude these types of buyers", "fair housing laws"],
    ["Only deal with certain demographics", "fair housing laws"],
    ["Steer away from those areas", "fair housing laws"],
    ["Not for those kinds of people", "fair housing laws"],
    ["Avoid these types of clients", "fair housing laws"],
    ["Prefer not to associate with them", "fair housing laws"],
    ["Exclude certain income levels", "fair housing laws"],
    ["Only cater to speciﬁc groups", "fair housing laws"],
    ["Steer clients to certain neighborhoods", "fair housing laws"],
    ["Not open to all families", "fair housing laws"],
    ["Avoid certain types of families", "fair housing laws"],
    ["Not interested in certain demographics", "fair housing laws"],
    ["Prefer homogenous neighborhood", "fair housing laws"],
    ["Selective about client race", "fair housing laws"],
    ["Gender-speciﬁc housing", "fair housing laws"],
    ["Excluding disabled tenants", "fair housing laws"],
    ["Age-restricted sales", "fair housing laws"],
    ["Ethnicity-based client choices", "fair housing laws"],
    ["Sexual orientation discrimination", "fair housing laws"],
    ["Family status limitations", "fair housing laws"],
    ["Biased housing advertisements", "fair housing laws"],
    ["Racially targeted marketing", "fair housing laws"],
    ["Exclusionary rental practices", "fair housing laws"],
    ["Segregation in housing offers", "fair housing laws"],
    ["Discriminatory lending practices", "fair housing laws"],
    ["Steering towards certain areas", "fair housing laws"],
    ["Blockbusting tactics", "fair housing laws"],
    ["Redlining practices", "fair housing laws"],
    ["Discriminatory zoning policies", "fair housing laws"],
    # Conflict of Interest and Disclosure Requirements
    ["Personal investment involved", "conflict of interest"],
    ["Hidden stake in the property", "conflict of interest"],
    ["Beneﬁtting from this deal", "conflict of interest"],
    ["My partner's listing", "conflict of interest"],
    ["Friend's property on sale", "conflict of interest"],
    ["Selling my own investment", "conflict of interest"],
    ["Family member's business", "conflict of interest"],
    ["Personal connection to the buyer", "conflict of interest"],
    ["lingo", "conflict of interest"],
    ["plonker", "bad words"],
    ["piss", "bad words"],
    ["hell", "wrong choice of words"],
    ["frustrations", "wrong choice of words"],
]

# Split recognized text into sentences
recognized_sentences = result["chunks"]

# Compute BM25 scores
avgdl = sum(len(sentence["text"]) for sentence in recognized_sentences) / len(
    recognized_sentences
)
N = len(recognized_sentences)

def bm25(word, sentence, k=1.2, b=0.75):
    freq = sentence["text"].count(word)
    tf = (freq * (k + 1)) / (freq + k * (1 - b + b * (len(sentence["text"]) / avgdl)))
    N_q = sum([doc["text"].count(word) for doc in recognized_sentences])
    idf = np.log(((N - N_q + 0.5) / (N_q + 0.5))) + 1
    return tf * idf

# Compute similarity scores
similarity_scores_with_categories = {}
for word, category in red_flag_data:
    scores = []
    for i, sentence in enumerate(recognized_sentences):
        score = bm25(word, sentence)
        if score > 0.0:  # Consider only scores above 0.0
            scores.append((i + 1, score, sentence["timestamp"][0], sentence["timestamp"][1]))
    if scores:
        similarity_scores_with_categories[word] = (category, scores)

# Initialize a list to store table rows
table_data = []

# Counter for serial number
serial_number = 1

# Populate table data
for word, (category, scores) in similarity_scores_with_categories.items():
    for doc_id, score, start_time, end_time in scores:
        table_data.append([serial_number, score, word, category, start_time, end_time])
        serial_number += 1

# Print the table
print(
    tabulate(
        table_data,
        headers=["Serial No.", "Similarity Score", "Red Flags", "Category", "Start Time", "End Time"],
    )
)