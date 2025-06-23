import matplotlib.pyplot as plt
import numpy as np
import soundfile as sf

if __name__ == '__main__':
    file_name='A1.wav'
    audio, sr = sf.read(f'data/{file_name}')
    t = np.arange(0, len(audio)) / sr

    plt.plot(t, audio)
    plt.show()