import librosa
import numpy as np
import matplotlib.pyplot as plt

padto = 50000

path = "sound/11111111111_04_02.dat"
y, sr = librosa.load(path)
l = len(y)
left_pad = (padto - l) // 2
righ_pad = padto - l - left_pad
y = np.pad(y, (left_pad, righ_pad), 'wrap')

# librosa.display.waveplot(y, sr=sr)

# pre_shape = y.shape

feat = librosa.stft(y, hop_length=512, n_fft=1024)
feat = np.abs(feat) ** 2
try:
    feat = librosa.feature.melspectrogram(S=feat, sr=sr, n_mels=128)
except:
    raise IOError(path)
#     import ipdb
#     ipdb.set_tracSe()
feat = librosa.power_to_db(feat, ref=np.max)
feat = feat[:96,:]
print(feat.shape)

plt.figure(figsize=(10, 8))  # 设置图像大小
plt.imshow(feat, aspect="auto", origin="lower", cmap='jet')
plt.colorbar()  # 显示颜色条
plt.title("Mel-Frequency Filter Banks")
plt.xlabel("Frame")
plt.ylabel("Mel Filter")
plt.show()


