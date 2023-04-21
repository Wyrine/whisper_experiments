from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import Audio, load_dataset
import torch

# load model and processor
processor = WhisperProcessor.from_pretrained("openai/whisper-medium")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-medium")

# load dummy dataset and read soundfiles
ds = load_dataset("common_voice", "fr", split="test", streaming=True)
ds = ds.cast_column("audio", Audio(sampling_rate=16_000))
input_speech = next(iter(ds))["audio"]["array"]
# tokenize
input_features = processor(input_speech, return_tensors="pt").input_features 
forced_decoder_ids = processor.get_decoder_prompt_ids(language = "fr", task = "translate")

predicted_ids = model.generate(input_features, forced_decoder_ids = forced_decoder_ids)
transcription = processor.batch_decode(predicted_ids, skip_special_tokens = True)
