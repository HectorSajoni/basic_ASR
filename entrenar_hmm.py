import os
import numpy as np
import preprocesamiento as pr
import joblib
from hmmlearn import hmm

def entrenar_hmm(carpeta_train='dataset/train', n_components=5):
    modelos = {}
    palabras = sorted(os.listdir(carpeta_train))
    for palabra in palabras:
        ruta_palabra = os.path.join(carpeta_train, palabra)
        if not os.path.isdir(ruta_palabra): continue
        secuencias = []
        longitudes = []
        for archivo in os.listdir(ruta_palabra):
            if archivo.endswith('.wav'):
                y = pr.preprocesar(os.path.join(ruta_palabra, archivo))
                mfcc = pr.extraer_mfcc(y)  # (frames, 13)
                secuencias.append(mfcc)
                longitudes.append(len(mfcc))
        X = np.vstack(secuencias)
        modelo = hmm.GaussianHMM(n_components=n_components, covariance_type='diag', n_iter=100)
        modelo.fit(X, lengths=longitudes)
        modelos[palabra] = modelo
    return modelos

def predict(ruta_audio, modelos):
    y = pr.preprocesar(ruta_audio)
    mfcc = pr.extraer_mfcc(y)
    puntuaciones = {}
    for palabra, modelo in modelos.items():
        puntuaciones[palabra] = modelo.score(mfcc)
    return max(puntuaciones, key=puntuaciones.get)

if __name__ == '__main__':
    hmm = entrenar_hmm()
    joblib.dump(hmm, 'modelos/modelos_hmm.joblib')