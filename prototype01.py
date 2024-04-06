from transformers import AutoTokenizer, AutoModel, Wav2Vec2Processor, Wav2Vec2ForCTC

import torch
import pyaudio
import numpy as np
import typing

def loadModel(wav2vec2_model_name, device):
    "Load the given model."
    wav2vec2_processor = Wav2Vec2Processor.from_pretrained(wav2vec2_model_name)
    wav2vec2_model = Wav2Vec2ForCTC.from_pretrained(wav2vec2_model_name).to(device)
    return wav2vec2_model, wav2vec2_processor

def recordAudio(time):
    "Record the audio, during the given time."
    try:
        # initialize pyaudio
        p = pyaudio.PyAudio()

        # open a new stream to record audio
        stream = p.open(format=pyaudio.paInt16, channels=1, rate=16000, input=True, frames_per_buffer=1024)

        # record n seconds of audio
        print("Recording...")
        frames = []
        for _ in range(0, int(16000 / 1024 * time)):
            data = stream.read(1024)
            frames.append(data)

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
        print("Empezando transcripcion...")
        f.write(transcription)
        print("Transcripcion hecha.")

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
        frames = recordAudio(10)
        speech = convertAudio(frames)

        transcription = modelPrediction(speech, wav2vec2_processor, wav2vec2_model, device)
        print(f"Has dicho: {transcription}")

        textTranscriptionTxt(transcription)

        try:
            option = int(input("¿Quiere volver a pasar voz a texto? Si=1, No=0."))
            if option == 1:
                print("Volviendo a grabar...")
            if option == 0:
                print("Cerrando programa.")
                condition = False

        except Exception as e:
            print(e)

if __name__ == '__main__':
    main()