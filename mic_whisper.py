import whisper
import math
import wave
import pyaudio
import sys
import datetime
import numpy as np

model = whisper.load_model("large")
options = whisper.DecodingOptions(fp16=False, language='en')

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = 1 if sys.platform == 'darwin' else 2
RATE = 44100
RECORD_SECONDS = 1

audio = pyaudio.PyAudio()
stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE, input=True, frames_per_buffer=CHUNK)

print('Recording...')
while True:
    data = stream.read(CHUNK)
    audio_data = np.frombuffer(data, dtype=np.float32)
    # audio_data = audio_data.astype(np.float32) / 32768.0
    print("transcribing...")
    result = model.transcribe(audio_data, language='en', verbose=True)
    print(datetime.datetime.now(), result['text'])




# load audio and calculate its duration
audio = whisper.load_audio("output.wav")
with wave.open("output.wav", 'r') as wav:
    sample_rate = wav.getframerate()
duration = len(audio) / sample_rate

# loop through the audio and process 10-second clips
for i in range(math.ceil(duration / 10)):
    # slice the audio to a 10-second clip
    clip_start = i * 10
    clip_end = min((i + 1) * 10, duration)
    # clip_audio = audio.extract_subsegment(clip_start, clip_end)
    clip_audio = audio[int(clip_start * sample_rate):int(clip_end * sample_rate)]


    # pad/trim the clip to fit 10 seconds exactly
    clip_audio = whisper.pad_or_trim(clip_audio, 10 * sample_rate)

    # make log-Mel spectrogram and move to the same device as the model
    mel = whisper.log_mel_spectrogram(clip_audio).to(model.device)

    # detect the spoken language
    _, probs = model.detect_language(mel)
    print(f"Detected language for clip {i+1}: {max(probs, key=probs.get)}")

    # decode the audio
    clip_audio = clip_audio.reshape(1, len(clip_audio), 1)

    result = whisper.decode(model, mel, options)

    # print the recognized text
    print(f"Recognized text for clip {i+1}: {result.text}")
