import os
import zipfile
import librosa
from torch.utils.data import Dataset
from tqdm import *
import numpy as np
import torch
import random
from mfcc import preemp

padding = 50000

class MFCCDataset(Dataset):
    def __init__(self, data_dir = 'voice'):
        self.data_dir = data_dir
        self.nn_labels = []
        self.nn_inputs = []
        self.get_melspec()
        self.data_num = len(self.nn_labels)
        self.nn_inputs = torch.tensor(self.nn_inputs)
        self.nn_labels = torch.tensor(self.nn_labels)
        self.ix = []


    def read_voice(self):
        zip_dir = '/home/tianqizhang/audio_recognize/data'  # 替换为你的.zip文件所在的文件夹路径
        # 设置.wav文件存放的目标文件夹路径

        # 确保目标文件夹存在
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)

        # 遍历指定文件夹下的所有文件
        for file_name in os.listdir(zip_dir):
            # 检查文件是否为.zip文件
            if file_name.endswith('.zip'):
                # 构建完整的文件路径
                file_path = os.path.join(zip_dir, file_name)
                # 使用zipfile模块打开.zip文件
                with zipfile.ZipFile(file_path, 'r') as zip_ref:
                    # 遍历.zip文件中的所有文件
                    for entry in zip_ref.namelist():
                        # 检查文件是否为.wav文件
                        if entry.endswith('.dat'):
                            # 从.zip文件中提取.wav文件
                            zip_ref.extract(entry, self.data_dir)

    def get_melspec(self):
        # 遍历目录中的所有.dat文件
        dat_files = []
        index = 1
        for root, dirs, files in os.walk(self.data_dir):
            files = sorted(files)
            if len(files)>1:
                last_file = files[-1]
                if len(last_file.split('_')) > 1:
                    if int(last_file.split('_')[1]) == 19:
                        index = 0
                    elif int(last_file.split('_')[1]) == 20:
                        index = 1
                    else:
                        print("incorrect filename")
                        exit(11)
                elif len(last_file.split('-')) > 1:
                    if int(last_file.split('-')[1]) == 19:
                        index = 0
                    elif int(last_file.split('-')[1]) == 20:
                        index = 1
                    else:
                        print("incorrect filename")
                        exit(11)
                else:
                    print("incorrect filename")
                    exit(11)
            for file in files:
                if file.endswith('.dat') or file.endswith('.dat'):
                    full_path = os.path.join(root, file)
                    dat_files.append(full_path)
                    if len(file.split('_')) > 1:
                        self.nn_labels.append(int(file.split('_')[1]) - index)
                    elif len(file.split('-')) > 1:
                        self.nn_labels.append((int(file.split('-')[1]) - index))

        print("loading dataset...")
        feature_cnt = 1
        for path in tqdm(dat_files):
            # 加载音频文件
            waveform, sample_rate = librosa.load(path)

            wave_len = len(waveform)
            waveform = preemp(waveform, u = 0.97)
            left_pad = (padding - wave_len) // 2
            right_pad = padding - wave_len - left_pad
            if left_pad>0:
                waveform = np.pad(waveform, (left_pad, right_pad), 'wrap')
            else:
                waveform = waveform[-left_pad:wave_len-1+right_pad]
            fbank = librosa.stft(waveform, hop_length=512, n_fft=1024)
            fbank = np.abs(fbank) ** 2
            fbank = librosa.feature.melspectrogram(S=fbank, sr=sample_rate, n_mels=128)
            fbank = librosa.power_to_db(fbank, ref=np.max)
            fbank = fbank[:96,:]
            fbank = fbank[np.newaxis,:]
            fbank = fbank.tolist()
            self.nn_inputs.append(fbank)
                # 打印MFCC系数的形状，以确认输出

    def set_ix(self, ix):
        self.ix = ix
        self.data_num = len(self.ix)

    def __getitem__(self, item):
        if len(self.ix):
            index = self.ix[item]
        else:
            index = item

        return self.nn_inputs[index], self.nn_labels[index]

    def __len__(self):
        return self.data_num


def random_crop(image):
    min_ratio = 0.5
    max_ratio = 0.8

    c, w, h = image.shape

    ratio = random.random()

    scale = min_ratio + ratio * (max_ratio - min_ratio)

    new_h = int(h * scale)
    new_w = int(w * scale)

    y = np.random.randint(0, h - new_h)
    x = np.random.randint(0, w - new_w)

    image = image[:,x:x+new_w, y:y+new_h]

    return image

