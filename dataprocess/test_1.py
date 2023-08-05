import math

import torch
import wave
import numpy as np
import librosa
path=r'D:\article\data\dataset\uttr-0000ba4811c943728c449840ec4b3693.pt'
a=torch.load(path)

import scipy.io.wavfile as wav
from python_speech_features import mfcc,delta

wav_file = r'D:\article\data\wav1\id10001\1zcIwhmdeo4\00001.wav'
fs,signal = wav.read(wav_file)


# def wav2data(filepath, num_mfcc=20, feature=10):
#     # 提取文件中的所有wav文件中的MFCC并将其转化为标准格式
#     path = filepath
#     length = np.array(os.listdir(path)).shape[0]
#     mfccs = np.zeros([length, num_mfcc * feature])  # 统一选用13个40维特征向量，如果没有13个则补零
#     for i in range(length):
#         file = path + "{0}.wav".format(i)
#         y, rate = librosa.load(file)
#         mfcc = librosa.feature.mfcc(y=y, sr=rate, n_mfcc=num_mfcc)
#         mfcc_1 = np.reshape(mfcc, [-1, ])
#         if mfcc_1.shape[0] >= num_mfcc * feature:
#             mfccs[i] = mfcc_1[:num_mfcc * feature]
#         else:
#             mfccs[i] = np.hstack((mfcc_1, np.zeros([num_mfcc * feature - mfcc_1.shape[0], ])))
# return

# def extract_mfcc(wav_arr):
#     fs=16000.0
#     mfcc_feat = mfcc(wav_arr, fs)
#     energy = np.sqrt(wav_arr)
#     mfcc_feat = np.stack((mfcc_feat, energy))
#     delta1 = delta(mfcc_feat, 1)
#     delta2 = delta(delta1, 1)
#     mfcc_feat = np.hstack((mfcc_feat, delta1, delta2))
#     return mfcc_feat.T

def get_mfcc(data):
    fs=16000.0
    wav_feature = mfcc(data, fs)
    print(wav_feature.shape)
    a=wav_feature.shape[0]
    energy = np.sqrt(np.abs(data))
    print(energy.shape)
    # 检查元素数量是否能够被 13 整除，如果不能整除则向数组末尾添加零
    remainder = energy.size % 13
    if remainder != 0:
        zeros_to_add = 13 - remainder
        energy = np.append(energy, np.zeros(zeros_to_add))

    # 使用 reshape() 函数将 b 升维到 13 维
    energy = np.reshape(energy, (-1, 13))
    energy = energy[:wav_feature.shape[0], :]
    # 检查结果
    print(energy.shape)
    feature = np.hstack((wav_feature, energy))
    d_mfcc_feat = delta(wav_feature, 1)
    d_mfcc_feat2 = delta(wav_feature, 2)
    feature = np.hstack((wav_feature, d_mfcc_feat, d_mfcc_feat2))
    return feature

feature_mfcc = get_mfcc(signal)

tensor_mfcc = torch.from_numpy(feature_mfcc)
print(tensor_mfcc.shape)
print(tensor_mfcc)