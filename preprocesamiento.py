import librosa
import numpy as np

def reduccion_ruido_espectral(y, sr, n_fft=1024, ruido_frames=5):
    # Estimar ruido de los primeros 'ruido_frames' frames
    S = librosa.stft(y, n_fft=n_fft, hop_length=512)
    mag = np.abs(S)
    ruido_mag = np.mean(mag[:, :ruido_frames], axis=1, keepdims=True)
    # Restar ruido (con suelo)
    mag_reducida = np.maximum(mag - ruido_mag, 0.01 * mag)
    # Reconstruir señal con fase original
    S_reducida = mag_reducida * np.exp(1j * np.angle(S))
    y_reducida = librosa.istft(S_reducida, hop_length=512)
    return y_reducida

def preprocesar(ruta, sr=16000):
    y, _ = librosa.load(ruta, sr=sr, mono=True)
    # Normalización a pico 1
    y = y / np.max(np.abs(y) + 1e-8)
    # Reducción de ruido
    y = reduccion_ruido_espectral(y, sr)
    # (Opcional) Recortar silencios al inicio/final
    y, _ = librosa.effects.trim(y, top_db=20)
    return y

def extraer_mfcc(y, sr=16000, n_mfcc=13):
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc, n_fft=1024, hop_length=512)
    # mfcc tendrá forma (n_mfcc, frames)
    return mfcc.T  # devolvemos (frames, 13) para manejarlo como secuencia
