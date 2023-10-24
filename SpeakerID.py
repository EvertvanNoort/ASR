import torch
from pyannote.audio import Pipeline

def Diarization(audio_path, rttm_path, model, token, num_speakers):

    pipeline = Pipeline.from_pretrained(model,use_auth_token = token)

    pipeline.to(torch.device("cuda:0"))

    print('Diarization started')

    # apply the pipeline to an audio file
    diarization = pipeline(audio_path, num_speakers = num_speakers)

    # dump the diarization output to disk using RTTM format
    with open(rttm_path, "w") as rttm:
        diarization.write_rttm(rttm)

    print('Diarization done')    
    pass

# audio_path = "/home/evert/Desktop/audio/1.mp3"
# rttm_path = "/home/evert/Desktop/audio/audio1.rttm"

# diarization_model = "pyannote/speaker-diarization-3.0"
# diarization_token = "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"

# Diarization(audio_path, rttm_path, diarization_model, diarization_token)
    