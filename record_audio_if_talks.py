import sounddevice as sd
import numpy as np
import soundfile as sf

# Variables para controlar si se está grabando o no y si el programa debe continuar ejecutándose
is_recording = False
continue_running = True
recording = []

def start_program(duration, threshold, sample_rate):
    global DURATION
    global THRESHOLD
    global SAMPLE_RATE
    DURATION = duration
    THRESHOLD = threshold
    SAMPLE_RATE = sample_rate

def start_recording():
    global is_recording
    print("Recording Started")
    is_recording = True

def stop_recording():
    global is_recording
    global continue_running
    print("Recording Finished")
    is_recording = False
    continue_running = False
    sd.stop()

def record_block(indata):
    global recording
    recording.append(indata.copy())

# Crea una función de callback que se llamará para cada bloque de audio
def callback(indata, frames, time, status):
    volume_norm = np.linalg.norm(indata) * 10
    print("|" * int(volume_norm))  # Imprime una barra de volumen
    if volume_norm > THRESHOLD and not is_recording:
        start_recording()
    elif volume_norm <= THRESHOLD and is_recording:
        stop_recording()
    if is_recording:
        record_block(indata)

# Abre el stream de audio
def open_audio_stream():
    with sd.InputStream(callback=callback, channels=1, samplerate=SAMPLE_RATE, blocksize=int(SAMPLE_RATE * DURATION)):
        while continue_running:
            pass
    # Concatena todos los bloques de audio
    recording_concat = np.concatenate(recording, axis=0)
    # Convierte el array de numpy a bytes
    recording_bytes = recording_concat.astype(np.int16).tobytes()
    return recording_bytes


# Inicia el programa con los valores deseados
if __name__=='__main__':
    import prototype01

    start_program(1, 10, 44100)
    r = open_audio_stream()

    t = prototype01.convertAudio(r)

    print(f'Resultados: {t}')
    # save_recording('recording.wav')
