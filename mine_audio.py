import soundfile
import numpy as np
import random


def loadWAV(filename, max_frames):
    
    max_audio = max_frames * 160 + 240
    # Read wav file and convert to torch tensor
    audio, sample_rate = soundfile.read(filename)

    audiosize = audio.shape[0]

    if audiosize <= max_audio:
        shortage = max_audio - audiosize + 1
        audio = np.pad(audio, (0, shortage), 'wrap')
        audiosize = audio.shape[0]

    startframe = np.array(
            [np.int64(random.random()*(audiosize-max_audio))])

    feats = []

    for asf in startframe:
        feats.append(audio[int(asf):int(asf)+max_audio])

    feat = np.stack(feats, axis=0).astype(np.float64)

    return feat


if __name__ == '__main__':
    speaker_1_s_1 = loadWAV('./data/1_00001.wav',
                            0)
    speaker_1_s_2 = loadWAV('./data/1_00002.wav',
                            0)
    print(speaker_1_s_1)
