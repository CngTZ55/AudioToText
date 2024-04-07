import pyaudio
import wave
import time
import numpy as np


chunk = 1024  # Tamaño del fragmento de audio en bytes
sample_rate = 44100  # Frecuencia de muestreo en Hz
format = pyaudio.paInt16  # Formato de audio
channels = 1  # Número de canales (1 para mono, 2 para estéreo)

def record_audio():
    p = pyaudio.PyAudio()  # Instancia de PyAudio

    stream = p.open(format=format,
                channels=channels,
                rate=sample_rate,
                input=True,
                frames_per_buffer=chunk)


    frames = []  # Lista para almacenar fragmentos de audio
    start_time = time.time()  # Tiempo de inicio de la grabación
    threshold = 10  # Umbral de detección de voz
    silence_duration = 1  # Duración mínima del silencio para detener la grabación

    while True:
        # Leer un fragmento de audio
        data = stream.read(chunk)

        # Convertir los datos de audio a formato NumPy
        data_numpy = np.frombuffer(data, dtype=np.int16)

        # Calcular la amplitud máxima del fragmento
        loudness = np.max(data_numpy)

        # Detectar si el nivel de audio supera el umbral
        if loudness > threshold:
            # Si hay sonido, comienza a grabar
            print('loudness > treshold')
            start_time = time.time()
            frames.append(data)
        
        # Detección de silencio
        # print(f'Start time: {start_time}')
        res = time.time() - start_time
        print(f'Actual time - Start time: {res}')
        if loudness < threshold and time.time() - start_time > silence_duration:
            # Si hay silencio durante un tiempo determinado, detiene la grabación
            break

        # Mostrar información de grabación en la consola (opcional)
        print(f"Loudness: {loudness}")

    stream.stop_stream()
    stream.close()
    p.terminate()

    return frames

# Convertir la lista de fragmentos en un array de NumPy
frames = record_audio()
if len(frames) > 0:
    audio_numpy = np.concatenate(frames)

    # Guardar el audio grabado en un archivo WAV
    wave_file = wave.open("grabacion.wav", 'wb')
    wave_file.setnchannels(channels)
    wave_file.setsampwidth(p.get_sample_size(format))
    wave_file.setframerate(sample_rate)
    wave_file.writeframes(audio_numpy)
    wave_file.close()

    print("Grabación finalizada.")

else:
    frames = record_audio()
