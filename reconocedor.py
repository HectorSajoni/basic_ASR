import sounddevice as sd
import time
import joblib
import entrenar_gmm
import entrenar_hmm
import soundfile as sf
#import os

print('Graba >> ')
time.sleep(0.2)

duration = 1
fs = 16000
sd.default.samplerate = fs
sd.default.channels = 1
myrecording = sd.rec(int(duration * fs))
sd.wait()

#sd.play(myrecording)
#sd.wait()

modelos_gmm = joblib.load('modelos/modelos_gmm.joblib')
modelos_hmm = joblib.load('modelos/modelos_hmm.joblib')
carpeta_test = 'dataset/test'

sf.write('mi_audio.wav', myrecording, fs)
prediccion_gmm = entrenar_gmm.predict('mi_audio.wav', modelos_gmm)
prediccion_hmm = entrenar_hmm.predict('mi_audio.wav', modelos_hmm)
#os.remove('mi_audio.wav')

print(f'Predicción GMM: {prediccion_gmm}')
print(f'Predicción HMM: {prediccion_hmm}')
