import joblib
import entrenar_gmm as ent
import os

modelos_gmm = joblib.load('modelos/modelos_gmm.joblib')
modelos_hmm = joblib.load('modelos/modelos_hmm.joblib')
carpeta_test = 'dataset/test'

def evaluar(modelos, carpeta_test, predecir_func):
    correctas = 0
    total = 0
    for palabra_real in os.listdir(carpeta_test):
        ruta_palabra = os.path.join(carpeta_test, palabra_real)
        if not os.path.isdir(ruta_palabra): continue
        for archivo in os.listdir(ruta_palabra):
            if archivo.endswith('.wav'):
                total += 1
                pred = predecir_func(os.path.join(ruta_palabra, archivo), modelos)
                if pred == palabra_real:
                    correctas += 1
    acc = correctas / total
    wer = 1 - acc
    return acc, wer

gmm_acc, gmm_wer = evaluar(modelos_gmm, carpeta_test, ent.predecir_gmm)
hmm_acc, hmm_wer = evaluar(modelos_hmm, carpeta_test, ent.predecir_hmm)

print(f'El accuracy de gmm es: {gmm_acc} y su WER es : {gmm_wer}')
print(f'El accuracy de hmm es: {hmm_acc} y su WER es : {hmm_wer}')