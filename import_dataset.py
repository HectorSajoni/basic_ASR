# Crea la carpeta dataset y guarda en /train y en /test las palabras.

import re
import os
import shutil

with open("dataset_original/testing_list.txt", "r", encoding="utf-8") as archivo:
    # Recorremos cada línea y le quitamos los espacios y saltos de línea en blanco
    testing_list = [linea.strip() for linea in archivo]

with open("dataset_original/validation_list.txt", "r", encoding="utf-8") as archivo:
    # Recorremos cada línea y le quitamos los espacios y saltos de línea en blanco
    validation_list = [linea.strip() for linea in archivo]

testing_list += validation_list
palabras = ['yes', 'no', 'up', 'down', 'right', 'left']

try:
    shutil.rmtree('dataset')
    print(f"La carpeta 'dataset' ha sido eliminada.")
except Exception as e:
    print(f"Error al eliminar: {e}")   

for palabra in palabras:
    os.makedirs(f"dataset/train/{palabra}", exist_ok=True)
    os.makedirs(f"dataset/test/{palabra}", exist_ok=True)
print(f"Las carpetas han sido creadas.")

def copiar():
    contador_copiados = 0

    # os.walk recorre la estructura de carpetas de forma recursiva (de arriba hacia abajo)
    for carpeta_actual, _, archivos in os.walk('dataset_original'):
        palabra = re.sub(r'dataset_original/', '', carpeta_actual)
        if palabra in ['dataset_original', '_background_noise_']:
            continue
        for archivo in archivos:
            ruta_completa_origen = os.path.join(carpeta_actual, archivo)
            destino = 'dataset/train'
            if palabra + '/' + archivo in testing_list:
                destino = 'dataset/test'
            destino += '/'+palabra

            ruta_completa_destino = os.path.join(destino, archivo)
            
            # Copiar el archivo
            shutil.copy2(ruta_completa_origen, ruta_completa_destino)
            print(f"Copiado: {ruta_completa_origen} -> {ruta_completa_destino}")
            contador_copiados += 1

    print(f"\nProceso terminado. Se copiaron {contador_copiados} archivos.")

if __name__ == "__main__":
    copiar()