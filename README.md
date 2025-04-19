# Audio a Texto
Se demuestra a continuación el uso de la IA para la conversión de audio a texto, 
concretamente mediante el modelo wav2vec2-large-xlsr-53-spanish de Facebook, y
*Finetuneado por jonatasgrosman*, el modelo puede se encontrado en: https://huggingface.co/jonatasgrosman/

## Notas Importantes
La precisión de este programa no es tan buena, pese a que el modelo reporta su precisión,
lo cierto es que cabe la posibilidad de que a ruidos y a entradas de audios pobres, la 
detección no sea la mejor. Considera el programa como una prueba de concepto más que
para un entorno real.

## ¿Cómo funciona?
Es un programa de línea de comandos básico, el cual al ejecutarlo cargará el modelo,
bien sea en la GPU (si se posee) o en la CPU, después, se pondrá el programa a 
escuchar por audios si se supera un volumen, registrándo este en un archivo txt,
y pregunta si se desea volver a escuchar.

### Instalación
1. Instalar las dependencias de requirements.txt con
```pip install -r requirements.txt```

*OPCIONAL: Se recomienda el uso de entornos virtuales con:*
```pip -m venv env```

```env\Scripts\activate``` Para activar entorno y después si instalar las dependencias.

2. Ejecutar el programa, y ¡Transcribir!
