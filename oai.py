# from pydub import AudioSegment
# import torch
# from transformers import WhisperProcessor, WhisperForConditionalGeneration, WhisperTokenizer

# # load model and processor
# processor = WhisperProcessor.from_pretrained("openai/whisper-medium")
# model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium")
# tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-medium")


# # load the audio file in m4a format
# audio = AudioSegment.from_file("/Users/kirolosshahat/Desktop/mom_stories_of_saints.m4a", format="m4a", frame_rate=16000)

# audio = audio[:10000]
# # tokenize
# input_features = processor(audio.raw_data, return_tensors="pt", sampling_rate=16000).input_features
# forced_decoder_ids = processor.get_decoder_prompt_ids(language = "en", task = "transliterate")

# # generate predictions
# predicted_ids = model.generate(input_features, forced_decoder_ids = forced_decoder_ids)
# transcription = processor.batch_decode(predicted_ids, skip_special_tokens = True)

# # # print the transcription
# # print(transcription)

# # # convert token ids to text
# # transcription = tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)

# # print the transcription
# print(transcription)



# from pydub import AudioSegment
# import torch
# from transformers import WhisperProcessor, WhisperForConditionalGeneration, WhisperTokenizer

# # load model, processor, and tokenizer
# processor = WhisperProcessor.from_pretrained("openai/whisper-medium")
# model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium")
# tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-medium")

# # load the audio file in m4a format
# audio = AudioSegment.from_file("/Users/kirolosshahat/Desktop/mom_stories_of_saints.m4a", format="m4a",frame_rate=16000)

# # tokenize
# input_features = processor(audio.raw_data, return_tensors="pt", sampling_rate=16000).input_features
# forced_decoder_ids = processor.get_decoder_prompt_ids(language = "en", task = "transliterate")

# # generate predictions
# predicted_ids = model.generate(input_features, forced_decoder_ids = forced_decoder_ids)

# # convert token ids to text
# transcription = tokenizer.batch_decode(predicted_ids, skip_special_tokens=True)

# # print the transcription
# print(transcription)

import numpy as np
import datasets
from pydub import AudioSegment
import torch
from transformers import WhisperProcessor, WhisperForConditionalGeneration, WhisperTokenizer

processor = WhisperProcessor.from_pretrained("openai/whisper-medium")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium")

# load the audio file in m4a format
audio = AudioSegment.from_file("/Users/kirolosshahat/Desktop/mom_stories_of_saints.m4a", format="m4a", frame_rate=16000)

# convert audio to a numpy array
audio_samples = np.array(audio.get_array_of_samples())

# create a dataset
# ds = datasets.Dataset.from_dict({'audio':audio_samples})
# ds = datasets.Dataset.from_tensors(
#     audio_samples,
#     {"sampling_rate": audio.frame_rate, "num_channels": audio.channels}
# )

# convert audio to a numpy array
audio_samples = np.array(audio.get_array_of_samples())
metadata = {"sampling_rate": audio.frame_rate, "num_channels": audio.channels}

# create a dataset object
data = {'audio': audio_samples, 'metadata': metadata}
ds = datasets.Dataset.from_dict(data)
ds = ds.cast_column("audio", datasets.Audio(sampling_rate=16_000))
input_speech = next(iter(ds))["audio"]["array"]

input_features = processor(input_speech, return_tensors="pt").input_features 
forced_decoder_ids = processor.get_decoder_prompt_ids(language = "en", task = "translate")

predicted_ids = model.generate(input_features, forced_decoder_ids = forced_decoder_ids)
transcription = processor.batch_decode(predicted_ids, skip_special_tokens = True)
print(transcription)

# # load model, processor, and tokenizer

# tokenizer = WhisperTokenizer.from_pretrained("openai/whisper-medium")

# # load the audio file in m4a format
# audio = AudioSegment.from_file("/Users/kirolosshahat/Desktop/mom_stories_of_saints.m4a", format="m4a",frame_rate=16000)

# # tokenize
# input_features = processor(audio.raw_data, return_tensors="pt", sampling_rate=16000).input_features
# forced_decoder_ids = processor.get_decoder_prompt_ids(language = "en", task = "transliterate")

# # generate predictions
# predicted_ids = model.generate(input_features, forced_decoder_ids = forced_decoder_ids)

# # convert token ids to text
# transcription = tokenizer.decode(predicted_ids[0],skip_special_tokens=True)

# # print the transcription
# print(transcription)
