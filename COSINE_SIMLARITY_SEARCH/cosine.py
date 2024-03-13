# Recognized text:  If there was one piece of advice you could give to EXP agents, what would that be? Entering your day with a lot of gratitude. I think that's the way I try to approach every day in life is to wake up grateful. You know, no matter where your market is, there's going to be someone that's going to need your help. there's going to be someone that's going to need your help. And, you know, you have to seek those people out. And you have to be able to show them a level of grace as well. Not everyone understands real estate. You know, we do it every day. So sometimes we just get into the motion, we just start talking our lingo, we start doing our thing and then the client is just confused. So I find that we have to kind of enter the day with that certain level of gratitude and talk to our clients with a little bit of grace. And I think that goes a long way.       
# Detected flag: gratitude
# Similarity Score: 0.08428995039221793
# Timestamp of flag: (30, 60)


import torch
import librosa
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

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
    return_timestamps=True,  # Make sure this parameter is set to True
    torch_dtype=torch_dtype,
    device=device,
)

# Load the audio file using librosa
input_audio, sample_rate = librosa.load('sound\podcast.mp3', sr=16000)

# Calculate timestamps manually based on the chunk indices and the sampling rate
chunk_duration = 30  # Duration of each chunk in seconds
timestamps = []
for i in range(len(input_audio) // (chunk_duration * sample_rate)):
    start_time = i * chunk_duration
    end_time = (i + 1) * chunk_duration
    timestamps.append((start_time, end_time))

# Perform speech recognition
result = pipe(input_audio)

# Print the keys in the result dictionary to inspect its structure
print(result.keys())

# Get the recognized text along with timestamps
transcription = result['text']

# Print the recognized text
print("Recognized text:", transcription)

# Sample red flags
red_flags = [
    "Totally unqualified",
    "She's a fraud",
    "They're useless",
    "Completely inept",
    "Total sham",
    "They're unethical",
    "What a disaster",
    "Unbelievably bad",
    "Totally shady",
    "They're a disgrace",
    "Absolute incompetence",
    "He's a cheat",
    "Such a scam",
    "Lacks integrity",
    "Totally unreliable",
    "Complete joke",
    "Unethical behavior",
    "Serious misconduct",
    "He's a charlatan",
    "They're corrupt",
    "Absolutely dishonest",
    "Lack of professionalism",
    "He's a swindler",
    "Totally unscrupulous",
    "She's deceitful",
    "They're fraudulent",
    "Completely unethical",
    "Utterly dishonest",
    "Blatantly corrupt",
    "Grossly incompetent",
    "Instant wealth creation",
    "Surefire investment success",
    "Quick financial gain",
    "Rapid wealth accumulation",
    "Easy profit-making",
    "Guaranteed return on investment",
    "Effortless financial growth",
    "Speedy income generation",
    "Rapid revenue increase",
    "Quick financial turnaround",
    "Instant money-making",
    "Guaranteed financial results",
    "Easy wealth accumulation",
    "Rapid profit-making",
    "Surefire wealth increase",
    "Effortless income boost",
    "Speedy profit increase",
    "Quick revenue growth",
    "Instant financial improvement",
    "Guaranteed wealth boost",
    "Easy financial success",
    "Rapid wealth gain",
    "Quick profit generation",
    "Instant revenue increase",
    "Guaranteed income boost",
    "Effortless wealth creation",
    "Speedy financial gains",
    "Quick wealth improvement",
    "Instant profit increase",
    "Guaranteed revenue boost",
    "Blasting calls out",
    "Spamming their inbox",
    "Bombarding with texts",
    "Nonstop robocalls",
    "Endless phone spam",
    "Persistent marketing tactics",
    "Secret profit sharing",
    "Covert referral fee",
    "Hidden kickback deal",
    "Undercovered commission",
    "Backhanded payment",
    "Disguised referral fee",
    "Cloaked compensation",
    "Sneaky profit split",
    "Concealed kickback",
    "Underhanded commission",
    "Shrouded payment",
    "Masked referral agreement",
    "Veiled kickback arrangement",
    "Hidden financial incentive",
    "Covert profit exchange",
    "Undercover referral agreement",
    "Sly commission deal",
    "Disguised profit sharing",
    "Cloaked financial reward",
    "Sneaky commission split",
    "Concealed referral incentive",
    "Underhanded profit arrangement",
    "Shrouded commission incentive",
    "Masked kickback deal",
    "Veiled financial agreement",
    "Hidden commission scheme",
    "Covert referral incentive",
    "Undercover profit sharing",
    "Sly kickback arrangement",
    "Disguised commission incentive",
    "Avoid renting to them",
    "Prefer not to sell to those people",
    "Don't deal with that community",
    "Exclude certain types",
    "Only rent to our kind",
    "Steer clear of that neighborhood",
    "Not suitable for certain families",
    "Don't want those kinds of tenants",
    "Avoid selling to that group",
    "Prefer not to work with them",
    "Exclude these types of buyers",
    "Only deal with certain demographics",
    "Steer away from those areas",
    "Not for those kinds of people",
    "Avoid certain types of clients",
    "Prefer not to associate with them",
    "Exclude certain income levels",
    "Only cater to specific groups",
    "Steer clients to certain neighborhoods",
    "Not open to all families",
    "Avoid certain types of families",
    "Not interested in certain demographics",
    "Prefer homogenous neighborhood",
    "Selective about client race",
    "Gender-specific housing",
    "Excluding disabled tenants",
    "Age-restricted sales",
    "Ethnicity-based client choices",
    "Religion-specific housing policies",
    "Sexual orientation discrimination",
    "Family status limitations",
    "Biased housing advertisements",
    "Racially targeted marketing",
    "Exclusionary rental practices",
    "Segregation in housing offers",
    "Discriminatory lending practices",
    "Steering towards certain areas",
    "Blockbusting tactics",
    "Redlining practices",
    "Discriminatory zoning policies",
    "lingo",
    "gratitude"
]

# Convert recognized text into vectors
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform([transcription])

# Initialize variables to store the flag, its similarity score, and the timestamp
detected_flag = None
max_similarity_score = 0.0
timestamp_of_flag = None

# Check for red flags in the recognized text
for red_flag in red_flags:
    # Check if the red flag is present in the recognized text
    if red_flag in transcription:
        # Calculate cosine similarity between the red flag and recognized text
        similarity_score = cosine_similarity(X, vectorizer.transform([red_flag]))[0, 0]
        # Update the detected flag and max similarity score if necessary
        if similarity_score > max_similarity_score:
            detected_flag = red_flag
            max_similarity_score = similarity_score
            # Find the position of the red flag in the transcription
            index_of_flag = transcription.find(red_flag)
            # Calculate the exact timestamp of the flag based on its position in the audio file
            for start, end in timestamps:
                if start <= index_of_flag / len(transcription) * (end_time - start_time) + start_time <= end:
                    timestamp_of_flag = (start, end)
                    break

# Print the detected flag, its similarity score, and the timestamp
if detected_flag:
    print(f"Detected flag: {detected_flag}")
    print(f"Similarity Score: {max_similarity_score}")
    print(f"Timestamp of flag: {timestamp_of_flag}")
else:
    print("No red flags detected.")
