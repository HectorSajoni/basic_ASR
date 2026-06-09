# basic_ASR

Este proyecto utiliza el dataset Google Speech Commands:
"Warden P. Speech Commands: A public dataset for single-word
speech recognition, 2017. Available from
http://download.tensorflow.org/data/speech_commands_v0.01.tar.gz".

Pasos de instalación:

1. Descargar el dataset, disponible en: "https://www.kaggle.com/datasets/neehakurelli/google-speech-commands/data" y descomprimirlo en una carpeta llamada "dataset_original" en la raíz del proyecto.

2. Ejecutar el módulo import_dataset.py para organizar los archivos.

3. Entrenar los modelos ejecutando los módulos entrenar_gmm.py y entrenar_hmm.py.

4. Ejecutar el módulo reconocedor.py, esperar a la instrucción de la señal y grabar el audio de un segundo.