import torch
import numpy as np
import librosa
from transformers import pipeline
from transformers import WhisperProcessor, WhisperForConditionalGeneration
from datasets import load_dataset, load_from_disk

def Transcribe(audio_path, rttm_path,model, output_path):
    # device = "cuda:0" if torch.cuda.is_available() else "cpu"
    device = torch.device('cuda:0')

    processor = WhisperProcessor.from_pretrained(model)
    model = WhisperForConditionalGeneration.from_pretrained(model).to(device)
    # processor.to(torch.device("cuda:0"))
    # model.to(torch.device("cuda:0"))
    model.config.forced_decoder_ids = None

    output_file = output_path
    output_file_simple = output_path

    # Read the RTTM file and extract speaker timestamps
    with open(rttm_path, 'r') as rttm_file:
        speaker_data = [line.strip().split() for line in rttm_file]

    # Initialize variables for speaker tracking
    transcriptions = []

    print('Starting transcription')

    for line in speaker_data:
        _, audio_file, _, start_time, duration, _, _, speaker, _, _ = line
        start_time = float(start_time)
        duration = float(duration)
        # speaker = speaker.to(device)

        # Load the audio segment and specify the sampling rate (16000 in this case)
        audio, sampling_rate = librosa.load(audio_path, sr=16000, offset=start_time, duration=duration)
        # audio = audio
        # sampling_rate = sampling_rate
        input_features = processor(audio, sampling_rate=sampling_rate, return_tensors="pt",max_new_tokens=4000).input_features
        predicted_ids = model.generate(input_features.to(device))
        transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
        text  = transcription[0]
        
        # Append the results to the transcription list
        transcriptions.append((start_time, speaker, text))

    # # Save the results to an output file
    # with open(output_file, 'w') as f1, open(output_file_simple, 'w') as f2, open(output_file_content, 'w') as f3:
    #     for start_time, speaker, transcription in transcriptions:
    #         timestamp_line = f"Timestamp: {start_time:.2f}s, Speaker: {speaker}, Transcription: {transcription}\n"
    #         simple_line = f"{speaker}: {transcription}\n"
    #         content_line = f"{transcription}\n"
            
    #         f1.write(timestamp_line)
    #         f2.write(simple_line)
    #         f3.write(content_line)

    # Save the results to an output file
    with open(output_file, 'w') as f1, open(output_file_simple, 'w') as f2:
        for start_time, speaker, transcription in transcriptions:
            timestamp_line = f"Timestamp: {start_time:.2f}s, Speaker: {speaker}, Transcription: {transcription}\n"
            simple_line = f"{speaker}: {transcription}\n"
            
            # f1.write(timestamp_line)
            f2.write(simple_line)

    print('Transcription done')
    pass