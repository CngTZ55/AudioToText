from transformers import AutoTokenizer, AutoModel, Wav2Vec2Processor, Wav2Vec2ForCTC
import record_audio_if_talks
import audioop

import torch
import pyaudio
import numpy as np
import typing
import sounddevice as sd
import time

def loadModel(wav2vec2_model_name, device):
    "Load the given model."
    wav2vec2_processor = Wav2Vec2Processor.from_pretrained(wav2vec2_model_name)
    wav2vec2_model = Wav2Vec2ForCTC.from_pretrained(wav2vec2_model_name).to(device)
    return wav2vec2_model, wav2vec2_processor

import sounddevice as sd
import numpy as np

# 
def recordAudio():
    "Record the audio, until silence is detected."
    try:
        # initialize pyaudio
        p = pyaudio.PyAudio()

        # open a new stream to record audio
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)

        # record audio until silence is detected
        print("Waiting for speech...")
        frames = []
        silence_threshold = 1000  # adjust this value based on your microphone sensitivity
        silence_count = 0
        speech_detected = False

        while True:
            data = stream.read(1024)
            rms = audioop.rms(data, 2)  # get the volume

            # if the volume is above the silence threshold, start recording
            if rms > silence_threshold:
                speech_detected = True
                frames.append(data)
                print("Recording...")
            elif speech_detected:
                # if the volume is below the silence threshold, increment the silence count
                if rms < silence_threshold:
                    silence_count += 1
                else:
                    silence_count = 0

                # if silence is detected for a certain amount of time, stop recording
                if silence_count > 100:  # adjust this value based on how long you want to wait before stopping the recording
                    break

        print("Done recording")

        # close the stream and pyaudio
        stream.stop_stream()
        stream.close()
        p.terminate()
        return frames
    except Exception as e:
        print(e)

def convertAudio(frames):
    "Convert the audio in to a format that can be treated by the model"
    # convert the recorded audio to a numpy array
    audio = np.frombuffer(b''.join(frames), dtype=np.int16)

    # convert the numpy array to a torch tensor
    speech = torch.from_numpy(audio.copy()).float()

    return speech
# def convertAudio(frames):
#     "Convert the audio in to a format that can be treated by the model"
#     # convert the recorded audio to a numpy array
#     audio = np.frombuffer(frames, dtype=np.int16)

#     # convert the numpy array to a torch tensor
#     speech = torch.from_numpy(audio.copy()).float()

#     return speech

def modelPrediction(speech, wav2vec2_processor, wav2vec2_model, device):
    "Using the given model for make predictions."
    # tokenize our wav
    input_values = wav2vec2_processor(speech, return_tensors="pt", sampling_rate=16000)["input_values"].to(device)
    input_values.shape

    # perform inference
    logits = wav2vec2_model(input_values)["logits"]
    logits.shape

    # use argmax to get the predicted IDs
    predicted_ids = torch.argmax(logits, dim=-1)
    predicted_ids.shape

    # decode the IDs to text
    transcription = wav2vec2_processor.decode(predicted_ids[0])
    transcription.lower()

    return transcription

def textTranscriptionTxt(transcription):
    "Just save in a file all the info you want."
    with open("transcription.txt", "w") as f:
        print("Writing...")
        f.write(transcription)
        print("Writing success!.")

def main():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"

    print("Loading the model...")

    model = "jonatasgrosman/wav2vec2-large-xlsr-53-spanish"
    wav2vec2_model, wav2vec2_processor = loadModel(model, device)

    #r = recordAudio(5)
    #print(r)
    #speech = convertAudio(r)
    #print(speech)

    #modelPrediction(speech, wav2vec2_processor, wav2vec2_model, device)

    condition = True
    while condition:
        frames = recordAudio()
        # record_audio_if_talks.start_program(1, 10, 44100)
        # frames = record_audio_if_talks.open_audio_stream()
        speech = convertAudio(frames)
        print(speech)

        transcription = modelPrediction(speech, wav2vec2_processor, wav2vec2_model, device)
        print(f"Has dicho: {transcription}")

        textTranscriptionTxt(transcription)

        try:
            option = int(input("Do you want to transform audio to text again? Yes=1, No=0."))
            if option == 1:
                print("Starting...")
            if option == 0:
                print("Closing program.")
                condition = False

        except Exception as e:
            print(e)

if __name__ == '__main__':
    main()