import librosa
import numpy as np

def extract_features(file_path):
    """
    Extract MFCC features from an audio file.
    Input: path to a wav file.
    Output: numpy array of mean MFCC features.
    """
    y, sr = librosa.load(file_path, sr=None)
    mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    mfccs_mean = np.mean(mfccs.T, axis=0)
    return mfccs_mean
