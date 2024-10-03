import scipy.io.wavfile as wav
import matplotlib.pyplot as plt
import cmath
import numpy as np
#from scipy.fftpack import dct
import pandas as pd

def fft(a):  #输入一个ndarray
    n = len(a)
    if n==1: #base case:只有一个元素的DFT是它自己
        return a
    w_n = cmath.exp(2*cmath.pi*1j/n) #旋转因子W_N
    w = 1
    a_0 = np.array([a[i] for i in range(0,n-1,2)]) #偶序列
    a_1 = np.array([a[i] for i in range(1,n,2)]) #奇序列
    y_0 = fft(a_0) #递归
    y_1 = fft(a_1) 
    y = np.zeros(n,dtype = complex) #创建合成数组
    for k in range(n//2): #通过奇偶序列计算原序列
        y[k] = y_0[k] + w*y_1[k]
        y[k+n//2] = y_0[k] - w*y_1[k]
        w = w*w_n #每次更新W为W_N^k
    return y

def preemp(signal, u=0.98): #预加重函数，u为预加重系数
    signal = signal[1:] - u*signal[:-1]
    return signal
    
def enframe(signal, frame_len, frame_step): #分帧
    signal_len = len(signal)
    if signal_len <= frame_len:
        numframes = 1
    else:
        numframes = 1 + int(np.ceil((signal_len - frame_len)/frame_step))
    padlen = int((numframes-1)*frame_step + frame_len)
    zeros = np.zeros((padlen - signal_len,))
    padsignal = np.concatenate((np.array(signal), zeros))
    indices = np.tile(np.arange(0, frame_len), (numframes, 1)) + np.tile(np.arange(0, numframes*frame_step, frame_step), (frame_len, 1)).T
    indices = np.array(indices, dtype=np.int32)
    frames = padsignal[indices]
    return frames

def add_window(frames, winfunc): #加窗
    win = np.tile(winfunc(len(frames[0])), (len(frames), 1))
    return frames * win

def vad_detect(signal, silence_len, plot = False):
    signal = np.array(signal)
    silence = signal[:silence_len]
    silence_energy = np.sum(np.square(silence), axis = 1)
    energy = np.sum(np.square(signal), axis = 1)
    silence_zrc = 0.5*(np.sum(np.abs(np.sign(silence[:,1:]) - np.sign(silence[:,:-1])), axis = 1))
    zrc = 0.5*(np.sum(np.abs(np.sign(signal[:,1:]) - np.sign(signal[:,:-1])), axis = 1))
    if plot:    
        plt.subplot(2,2,3)
        plt.plot(energy)
        plt.subplot(2,2,4)
        plt.plot(zrc)
    energy_mean = silence_energy.mean()
    zrc_mean = silence_zrc.mean()
    MH = energy_mean*60
    ML = energy_mean*30
    index = [ix for ix, value in enumerate(energy) if value > MH]
    N1 = min(index)
    N2 = max(index)
    index_2 = [ix for ix, value in enumerate(energy) if value > ML and (ix < N1 or ix > N2)]
    N1_2 = min(index_2) if min(index_2) < N1 else N1
    N2_2 = max(index_2) if max(index_2) > N2 else N2
    start  = 0
    end = 0
    for i in range(N1_2,N1_2-int(0.2/(win_len/rt)),-1):
        if zrc[i] < 4*zrc_mean:
            start = i
            break
    print(200*(win_len/rt))
    for i in range(N2_2, N2_2+int(0.2/(win_len/rt)), 1):
        if zrc[i] < 4*zrc_mean:
            end = i
            break
    print(start, N1_2, N1, N2, N2_2, end)
    return start, N1_2, N1, N2, N2_2, end

def fft_spectrum(frames, FFT): #对加窗后序列做fft变换
    spectrums = []
    Length = np.array(frames).shape[1]
    for frame in frames:
        spectrum = FFT(frame)[:Length//2+1]
        spectrums.append(spectrum)
    return spectrums
    
def plot_wav(wavsignal, rt): #画出波形图
    time = np.arange(0, len(wavsignal)) * (1.0 / rt)
    plt.plot(time, wavsignal)
    plt.xlabel('Time(s)')
    plt.ylabel('Amplitude')
    plt.title('Waveform')
    
def plot_wav_start_end(wavsignal, rt, start, N1_2, N1, N2, N2_2, end):
    time = np.arange(0, len(wavsignal)) * (1.0 / rt)
    plt.plot(time, wavsignal)
    plt.axvline(x = start/rt, color = 'r',linestyle='--')
    plt.axvline(x = end/rt, color = 'r', linestyle='--')
    plt.xlabel('Time(s)')
    plt.ylabel('Amplitude')
    plt.title('Waveform')

def plot_vad(signal, rt, wavsignal, win_len, plot = False):
    start, N1_2, N1, N2, N2_2, end = vad_detect(signal, 20, plot)
    if plot:
        plt.subplot(2,2,1)
        plot_wav(wavsignal, rt)
        plt.subplot(2,2,2)
        plot_wav_start_end(wavsignal, rt, start*win_len/2, N1_2*win_len/2, N1*win_len/2, N2*win_len/2, N2_2*win_len/2, end*win_len/2)
        plt.show()
    return start, end
    
def mfcc(signal, rt, L, N, D):
    signal = preemp(signal, 0.97)
    signal = enframe(signal, N, N/2)
    signal = add_window(signal, np.hamming)
    signal = fft_spectrum(signal, fft)
    signal = signal[:L] 
    signal = np.square(np.abs(signal))
    low_freq_mel = 0
    high_freq_mel = (2595 * np.log10(1 + (rt / 2) / 700))
    mel_points = np.linspace(low_freq_mel, high_freq_mel, D + 2)  # Equally spaced in Mel scale
    hz_points = (700 * (10**(mel_points / 2595) - 1))  # Convert Mel to Hz
    
    bin = np.floor((N + 1) * hz_points / rt)
    filters = np.zeros((D, N //2 + 1))
    for m in range(1, D + 1):
        f_m_minus = int(bin[m - 1])   # left
        f_m = int(bin[m])             # center
        f_m_plus = int(bin[m + 1])    # right
        for k in range(f_m_minus, f_m):
            filters[m - 1, k] = (k - bin[m - 1]) / (bin[m] - bin[m - 1])
        for k in range(f_m, f_m_plus):
            filters[m - 1, k] = (bin[m + 1] - k) / (bin[m + 1] - bin[m])   
    mfcc_result = np.dot(signal, filters.T)
    mfcc_result = 20 * np.log10(mfcc_result)  # dB
    return mfcc_result

def get_mfcc(file_path):
    rt, wavsignal = wav.read(file_path)
    win_len = 64
    signal = preemp(wavsignal, 0.97)
    signal = enframe(signal, win_len, win_len/2)
    signal = add_window(signal, np.hamming)
    start, end = plot_vad(signal, rt, wavsignal, win_len, True)
    signal = wavsignal[start*win_len:end*win_len]
    mfcc_coef = mfcc(signal, rt, 25, 512, 26)
    mfcc_coef -= mfcc_coef.mean(axis = 0) + 1e-8  
    mfcc = dct(mfcc_coef, type=2, axis=1, norm='ortho')[:,1:17]   
    mfcc -= mfcc.mean(axis = 0) + 1e-8
    df = pd.DataFrame(mfcc)
    df.to_csv("mfcc.csv", index = False)

if __name__ == '__main__':
    rt, wavsignal = wav.read('sound/11111111111_00_00.dat')
    win_len = 64
    signal = preemp(wavsignal, 0.97)
    signal = enframe(signal, win_len, win_len/2)
    signal = add_window(signal, np.hamming)
    start, end = plot_vad(signal, rt, wavsignal, win_len, True)
    signal = wavsignal[start*win_len:end*win_len]
    

