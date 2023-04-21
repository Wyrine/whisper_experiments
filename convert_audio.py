from pydub import AudioSegment
import numpy as np
import wave
import openai
import json

openai.api_key = "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"

def convert_m4a_to(filename):
    # Load the M4A file
    m4a_file = AudioSegment.from_file(filename, format="m4a")

    # Convert the audio file to a waveform array
    waveform = np.array(m4a_file.get_array_of_samples())

    # Get the sample rate of the audio file
    sample_rate = m4a_file.frame_rate
    
    return waveform, sample_rate


def convert_wav_to(filename):
    wave_file = wave.open(filename)

    # Extract the waveform data from the audio file
    num_frames = wave_file.getnframes()
    waveform = np.fromstring(wave_file.readframes(num_frames), dtype=np.int16)
    
    # Close the wave file
    wave_file.close()
    return waveform, 16000

def transcribe_multilingual(waveform, sample_rate):
    # Set the model and prompt
    model = "transformer/whisper-multilingual-speech-to-text" 
    model = "transformer/whisper-base"
    prompt = "Transcribe this Arabic audio to English text"
    # Convert the NumPy array to a list
    data = waveform.tolist()

    # Convert the list to a JSON-serializable dictionary
    data = {"waveform": data}

    # Convert the dictionary to a JSON string
    data = json.dumps(data)

    # Use the OpenAI API to transcribe the waveform data
    return openai.Completion.create(
        engine=model,
        prompt=prompt,
        temperature=0.5,
        max_tokens=2048,
        top_p=1,
        frequency_penalty=0,
        presence_penalty=0,
        # waveform=waveform,
        data=data,
        sample_rate=sample_rate,
    )

if __name__ == "__main__":
    waveform, sample_rate = convert_m4a_to("/Users/kirolosshahat/Desktop/mom_stories_of_saints.m4a")
    response = transcribe_multilingual(waveform, sample_rate)
    # Print the transcribed text
    print(response["choices"][0]["text"])