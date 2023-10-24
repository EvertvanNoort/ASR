from SpeakerID import Diarization
from TranscribeFromID2 import Transcribe
import os
import fileinput

# audio_path = "/home/evert/Desktop/audio/1.mp3"
audio_path = '/home/evert/Desktop/audio/Ludenbos.mp3'
rttm_path = "/home/evert/Desktop/audio/audio1.rttm"
output_path = "/home/evert/Desktop/audio/myOutput.txt"

diarization_model = "pyannote/speaker-diarization-3.0"
diarization_token = "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"

transcription_model = "openai/whisper-medium"

num_speakers = 2

Diarization(audio_path, rttm_path, diarization_model, diarization_token, num_speakers)
Transcribe(audio_path, rttm_path, transcription_model, output_path)

# x = "SPEAKER_00"
# y = "Beth"
 
# for l in fileinput.input(files = output_path):
#     l = l.replace(x, y)
#     sys.stdout.write(l)

# def Name(output_path, target, name):
# 	with open(output_path, 'w') as f:
# 		lines = f.readlines()
# 		# print(lines)
# 		for line in lines:
# 			modline = line.replace("SPEAKER_00", "Beth")
# 			f.write(modline)

# Name(output_path, "SPEAKER_00", "Beth")