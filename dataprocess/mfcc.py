import json
import torch
import librosa
import numpy as np
import scipy.io.wavfile as wav
import glob
import uuid
import os
import random

base_dir = r'D:\article\data\dataset\wav1'
save_dir = r"D:\article\data\dataset"

all_json_dir = r"D:\article\data\dataset\alldata.json"
meta_json_dir = r"D:\article\data\dataset\metadata.json"
test_json_dir = r"D:\article\data\dataset\testdata.json"


def get_mfcc(data):
    fs = 16000.0
    # wav_feature =  mfcc(data, fs)
    # d_mfcc_feat = delta(wav_feature, 1)
    # d_mfcc_feat2 = delta(wav_feature, 2)
    # feature = np.hstack((wav_feature, d_mfcc_feat, d_mfcc_feat2))
    mfccs = librosa.feature.mfcc(y=data, sr=fs, n_mfcc=40)
    mfccs = np.transpose(mfccs)
    mfccs -= (np.mean(mfccs, axis=0) + 1e-8)
    return mfccs


def data_write(wav_file):
    fs, signal = wav.read(wav_file)
    signal = signal.astype('float')
    feature = get_mfcc(signal)
    tensor = torch.from_numpy(feature)
    # 生成随机的文件名
    file_name = 'uttr-' + uuid.uuid4().hex + '.pt'
    # 拼接完整的保存路径和文件名
    save_path = os.path.join(save_dir, file_name)
    all_data['speakers'][id_dir].append({
        'feature_path': file_name,
        'mel_len': tensor.shape[0]
    })
    meta_data['speakers'][id_dir].append({
        'feature_path': file_name,
        'mel_len': tensor.shape[0]
    })
    torch.save(tensor, save_path)


def data_write_test(wav_file):
    fs, signal = wav.read(wav_file)
    signal = signal.astype('float')
    feature = get_mfcc(signal)
    tensor = torch.from_numpy(feature)
    # 生成随机的文件名
    file_name = 'uttr-' + uuid.uuid4().hex + '.pt'
    # 拼接完整的保存路径和文件名
    save_path = os.path.join(save_dir, file_name)
    all_data['speakers'][id_dir].append({
        'feature_path': file_name,
        'mel_len': tensor.shape[0]
    })
    test_data['utterances'].append({
        'feature_path': file_name,
        'mel_len': tensor.shape[0]
    })
    torch.save(tensor, save_path)


all_data = {}
all_data['n_mels'] = 40
all_data['speakers'] = {}
meta_data = {}
meta_data['n_mels'] = 40
meta_data['speakers'] = {}
test_data = {}
test_data['n_mels'] = 40
test_data['utterances'] = []

random_int = []
random_int = random.sample(range(10001, 11251 + 1), 500)
print(type(random_int))
random_sort = sorted(random_int)
print(random_sort)

for id_dir in os.listdir(base_dir):
    # 提取所有wav
    # 遍历每个 id 目录
    if not os.path.isdir(os.path.join(base_dir, id_dir)):
        continue
    all_data['speakers'][id_dir] = []
    meta_data['speakers'][id_dir] = []
    # test_data['speakers'][id_dir] = []
    # 判断当前 id 是否在指定的范围内
    id_number = int(id_dir[2:])
    if id_number not in random_sort:
        for sub_dir in os.listdir(os.path.join(base_dir, id_dir)):
            sub_dir_path = os.path.join(base_dir, id_dir, sub_dir)
            if not os.path.isdir(os.path.join(base_dir, id_dir, sub_dir)):
                continue
            # 使用 glob.glob 函数获取子目录中所有的 wav 文件
            wav_files_1 = glob.glob(os.path.join(base_dir, id_dir, sub_dir, '*.wav'))
            # 处理这些 wav 文件，例如读取和处理音频数据等
            for wav_file_01 in wav_files_1:
                data_write(wav_file_01)
    else:
        # 遍历 id 目录下所有子目录
        for sub_dir_1 in os.listdir(os.path.join(base_dir, id_dir)):
            sub_dir_path_1 = os.path.join(base_dir, id_dir, sub_dir_1)
            if not os.path.isdir(os.path.join(base_dir, id_dir, sub_dir_1)):
                continue
            # 获取当前子文件夹中所有的 wav 文件，并按照名称排序
            wav_files_2 = glob.glob(os.path.join(base_dir, id_dir, sub_dir_1, '*.wav'))
            if len(wav_files_2) == 1:
                data_write(wav_files_2)
            # 如果子文件夹中有多个文件，则提取除了最后一个 wav 文件之外的其它所有文件
            elif len(wav_files_2) > 1:
                if len(wav_files_2) == 2:
                    for wav_two in wav_files_2:
                        data_write(wav_two)
                else:
                    target_wavs = wav_files_2[:-1]
                    last_wavs = wav_files_2[-1]
                    for target_wav in target_wavs:
                        data_write(target_wav)
                    data_write_test(last_wavs)

# 将数据结构保存到 json 文件
with open(all_json_dir, 'w') as f:
    json.dump(all_data, f, indent=4)

with open(meta_json_dir, 'w') as f:
    json.dump(meta_data, f, indent=4)

with open(test_json_dir, 'w') as f:
    json.dump(test_data, f, indent=4)
