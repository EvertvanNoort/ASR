from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset, load_from_disk
import librosa

# load model and processor
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
model.config.forced_decoder_ids = None

# load dummy dataset and read audio files
# ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
audio, sampling_rate = librosa.load('/home/evert/Desktop/audio/1.mp3', sr = 16000)
# sample = ds[70]["audio"]
sample = audio
# input_features = processor(sample["array"], sampling_rate=sample["sampling_rate"], return_tensors="pt").input_features 
input_features = processor(sample, sampling_rate=sampling_rate, return_tensors="pt").input_features 

# generate token ids
predicted_ids = model.generate(input_features)
# decode token ids to text
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=False)
# ['<|startoftranscript|><|en|><|transcribe|><|notimestamps|> Mr. Quilter is the apostle of the middle classes and we are glad to welcome his gospel.<|endoftext|>']

transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True) 	
print(transcription)