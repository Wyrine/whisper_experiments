import datetime
import threading
import queue
import traceback
import speech_recognition as sr
from whisper_jax import FlaxWhisperPipline
import jax.numpy as jnp
import pytube


def transcribe_audio(model, audio_data):
    result = model.transcribe(audio_data, language='en', verbose=True)
    print(datetime.datetime.now(), result['text'])
    return result['text']

def record_audio_to_buffer_new(buffer, stop_event):
    print("starting to listen...")
    video = 'https://youtu.be/9eRyhmkzkLs'
    data = pytube.YouTube(video)
    # Converting and downloading as 'MP4' file

    audio = data.streams.get_audio_only()
    path = audio.download(filename=f'{data.title}.mp4')
    buffer.put(path)
    r = sr.Recognizer()
    with sr.Microphone() as source:
        while not stop_event.is_set():
            audio_data = r.listen(source)
            buffer.put(audio_data.get_wav_data())
            print("added some audio data to buffer")
    print("terminated listening")

def transcribe_buffered_audio(buffer, stop_event):
    # instantiate pipeline
    pipeline = FlaxWhisperPipline("openai/whisper-small", batch_size=16, dtype=jnp.bfloat16)
    while not stop_event.is_set():
        try:
            if not buffer.empty():
                audio_data = buffer.get()
                text = pipeline(audio_data, task='translate')
                print(text)
        except:
            traceback.print_exc()
    print("Done with", "transcribe_buffered_audio")

if __name__ == '__main__':
    # Create the buffer for audio data
    audio_buffer = queue.Queue()
    # Create an event to signal the threads to stop
    stop_event = threading.Event()
    try:
        # Create and start the recording thread
        record_thread = threading.Thread(target=record_audio_to_buffer_new, args=(audio_buffer, stop_event))
        record_thread.start()

        # Create and start the transcription thread
        transcribe_thread = threading.Thread(target=transcribe_buffered_audio, args=(audio_buffer, stop_event))
        transcribe_thread.start()

        # Wait for the threads to finish (which they won't, unless an error occurs or KeyboardInterrupt is raised)
        record_thread.join()
        transcribe_thread.join()
    except KeyboardInterrupt:
        print("\nStopping the program...")
    finally:
        # Set the stop event to signal the threads to exit
        stop_event.set()

        # Wait for the threads to finish
        record_thread.join()
        transcribe_thread.join()


