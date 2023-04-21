from pydub import AudioSegment
import whisper

# Load the M4A file
audio_segment = AudioSegment.from_file("/Users/kirolosshahat/Desktop/mom_stories_of_saints.m4a", format="m4a")
model = whisper.load_model("base")
# Extract the first 10 seconds of the audio
first_10_seconds = audio_segment[:10000]
result = model.transcribe(first_10_seconds)
print(result["text"])

# # Initialize the whisper client
# client = whisper.Client(api_key="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")

# # Transcribe the audio file
# text = client.transcribe(first_10_seconds)

# print(text)
