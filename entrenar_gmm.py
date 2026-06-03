from sklearn.mixture import GaussianMixture
import os
import numpy as np
import preprocesamiento as pr
import joblib

def entrenar_gmm(carpeta_train='dataset/train', n_components=8):
    modelos = {}
    palabras = sorted(os.listdir(carpeta_train))
    for palabra in palabras:
        ruta_palabra = os.path.join(carpeta_train, palabra)
        if not os.path.isdir(ruta_palabra): continue
        datos_palabra = []
        for archivo in os.listdir(ruta_palabra):
            if archivo.endswith('.wav'):
                y = pr.preprocesar(os.path.join(ruta_palabra, archivo))
                mfcc = pr.extraer_mfcc(y)
                datos_palabra.append(mfcc)  # cada uno es (frames, 13)
        # concatenar todos los frames de la palabra
        X = np.vstack(datos_palabra)  # (total_frames, 13)
        gmm = GaussianMixture(n_components=n_components, covariance_type='diag', max_iter=200)
        gmm.fit(X)
        modelos[palabra] = gmm
    return modelos

def predict(ruta_audio, modelos):
    y = pr.preprocesar(ruta_audio)
    mfcc = pr.extraer_mfcc(y)  # (frames, 13)
    puntuaciones = {}
    for palabra, gmm in modelos.items():
        # log‑likelihood de toda la secuencia
        loglike = gmm.score(mfcc)
        # score() ya suma los log‑probs de cada frame
        puntuaciones[palabra] = loglike
    return max(puntuaciones, key=puntuaciones.get)

if __name__ == '__main__':
    gmm = entrenar_gmm()
    joblib.dump(gmm, 'modelos/modelos_gmm.joblib')