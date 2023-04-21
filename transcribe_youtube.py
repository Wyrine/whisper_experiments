import whisper
import pytube

video = 'https://www.youtube.com/watch?v=nMbvbSqOPcU&ab_channel=StMercurius%26StAbraamCopticOrthodoxChurch'
data = pytube.YouTube(video)
# Converting and downloading as 'MP4' file
audio = data.streams.get_audio_only()
path = audio.download(filename=f'./{data.title}.mp4')

model = whisper.load_model("large")
result = model.transcribe(path, verbose=True, **{'language':'en'})
print(result["text"])