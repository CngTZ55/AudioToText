import pyaudio
import audioop

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
                if silence_count > 30:  # adjust this value based on how long you want to wait before stopping the recording
                    break

        print("Done recording")

        # close the stream and pyaudio
        stream.stop_stream()
        stream.close()
        p.terminate()
        return frames
    except Exception as e:
        print(e)

# Call the function to start recording
r = recordAudio()
print(r)
