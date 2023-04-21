import numpy as np
import whisper
import datetime
import threading
import queue
import traceback
import speech_recognition as sr
from speechbrain.pretrained import EncoderClassifier
import torch

def transcribe_audio(model, audio_data):
    result = model.transcribe(audio_data, language='en', verbose=True)
    print(datetime.datetime.now(), result['text'])
    return result['text']

def record_audio_to_buffer_new(buffer, stop_event):
    print("starting to listen...")
    r = sr.Recognizer()
    with sr.Microphone() as source:
        while not stop_event.is_set():
            audio_data = r.listen(source)
            buffer.put(audio_data.get_wav_data())
            print("added some audio data to buffer")

    print("terminated listening")

def determine_language(model, input_buffer, output_buffer, stop_event):
    language_id = EncoderClassifier.from_hparams(source="speechbrain/lang-id-voxlingua107-ecapa", savedir="tmp")
    while not stop_event.is_set():
        if not input_buffer.empty():
            audio_data = np.frombuffer(input_buffer.get(), dtype=np.float32)
            # padded_audio = whisper.pad_or_trim(audio_data)
            
            # normalized_audio = padded_audio.astype(np.float32) / 32768.0
            # # # make log-Mel spectrogram and move to the same device as the model
            # mel = whisper.log_mel_spectrogram(normalized_audio).to(model.device)

            # detect the spoken language
            # _, probs = model.detect_language(mel)
            # language = max(probs, key=probs.get)
            
            language = language_id.classify_batch(torch.from_numpy(audio_data))[3]
            print("detected", language)
            output_buffer.put((audio_data, language[0][:2]))
            # output_buffer.put((audio_data, 'en'))
    print("Done with", "determine_language")

def transcribe_buffered_audio(model, buffer, stop_event):
    
    while not stop_event.is_set():
        try:
            if not buffer.empty():
                audio_data, language = buffer.get()
                print("language in transcribe_buffered_audio", language)
                audio_data = audio_data.astype(np.float32) / np.iinfo(np.int16).max
                text = transcribe_audio(model, audio_data)
        except:
            traceback.print_exc()
    print("Done with", "transcribe_buffered_audio")

if __name__ == '__main__':
    # Create the buffer for audio data
    audio_buffer = queue.Queue()
    # Create a buffer for audio data as well as the associated language
    audio_and_language_buffer = queue.Queue()
    # Create an event to signal the threads to stop
    stop_event = threading.Event()
    model = whisper.load_model("small")
    try:
        # Create and start the recording thread
        record_thread = threading.Thread(target=record_audio_to_buffer_new, args=(audio_buffer, stop_event))
        record_thread.start()

        language_thread = threading.Thread(target=determine_language, args=(model, audio_buffer, audio_and_language_buffer, stop_event))
        language_thread.start()
        # Create and start the transcription thread
        transcribe_thread = threading.Thread(target=transcribe_buffered_audio, args=(model, audio_and_language_buffer, stop_event))
        transcribe_thread.start()

        # Wait for the threads to finish (which they won't, unless an error occurs or KeyboardInterrupt is raised)
        record_thread.join()
        language_thread.join()
        transcribe_thread.join()

    except KeyboardInterrupt:
        print("\nStopping the program...")
    finally:
        # Set the stop event to signal the threads to exit
        stop_event.set()

        # Wait for the threads to finish
        record_thread.join()
        language_thread.join()
        transcribe_thread.join()


