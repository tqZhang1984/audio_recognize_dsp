import torch
from torch.utils.data import DataLoader, random_split
from model import ResNet18
from dataset import MFCCDataset
from torch.nn import CrossEntropyLoss
from torch import optim
from tqdm import *
from tensorboardX import SummaryWriter
from sklearn.model_selection import StratifiedKFold
import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

configs = {
    'batch_size':4096,  # 超参数batch大小
    'epoch':300,  # 一折的训练轮数
    'save_dir':"./audio_ResNet_V5_",  # 模型权重参数保存位置
    'lr':3e-5,
    'log_dir':'runs/audio_recog_V5'
}

result = ["数字", "语音","语言","处理","中国", "忠告", "北京", "背景", "上海", "商行",
          "Speech","Speaker", "Signal","Sequence","Processing","Print","Project","File","Open","Close"]


class Runner:
    def __init__(self, **kwargs):
        # 设定参数
        self.device = torch.device("cuda:0")
        self.save_dir = kwargs['save_dir']
        self.batch_size = kwargs['batch_size']
        self.total_iter = kwargs['epoch']
        self.lr = kwargs['lr']
        self.log_dir = kwargs['log_dir']


        self.audio_network = ResNet18(classes_num=20).to(self.device)
        params = self.audio_network.parameters()
        self.optimizer = optim.Adam(params, lr=self.lr, betas=(0.9,0.999), weight_decay=5e-4)
        self.loss_func = CrossEntropyLoss().to(self.device)
        self.current_iter = 0
        self.writer = SummaryWriter(self.log_dir)
        self.train_dataset = None
        self.test_dataset = None
        self.train_iter = None
        self.test_iter = None
        self.n_split = 5

    def load_dataset(self, data_dir):
        self.train_dataset = MFCCDataset(data_dir)
        self.test_dataset = MFCCDataset("test_voice")

    def train(self):
        best_acc = 0
        skf = StratifiedKFold(n_splits=self.n_split)
        cnt = 0
        acc_hist = 0
        for i, (train_ix, test_ix) in enumerate(skf.split(self.train_dataset.nn_inputs, self.train_dataset.nn_labels)):
            self.current_iter = 0
            self.train_dataset.set_ix(train_ix)
            self.train_iter = DataLoader(self.train_dataset, batch_size=self.batch_size,
                                         shuffle = True)
            self.test_iter = DataLoader(self.test_dataset, batch_size=self.batch_size,
                                        shuffle = False)
            with tqdm(total=self.total_iter,
                          desc=f"Iteration {self.current_iter}/{self.total_iter}") as pbar:
                while self.current_iter <= self.total_iter:
                    self.audio_network.train()
                    for train_input, train_label in self.train_iter:
                        if self.current_iter > self.total_iter*self.n_split:
                            break

                        train_input, train_label = train_input.to(self.device), train_label.to(self.device)
                        self.optimizer.zero_grad()
                        output = self.audio_network(train_input)
                        loss = self.loss_func(output, train_label)
                        loss.backward()
                        self.optimizer.step()
                        self.current_iter += 1

                        pbar.update(1)
                        pbar.set_description(f"Iteration {self.current_iter}/{self.total_iter}")
                        pbar.set_postfix_str('loss = {:.6f}, lr = {:.6f}'.format(loss.item(), self.optimizer.param_groups[0]['lr']))
                        self.writer.add_scalar('Loss/train', loss, self.current_iter+i*self.total_iter)


                        if self.current_iter % 10 == 0:
                            test_acc = self.evaluate(i)
                            if test_acc > 0.98:
                                acc_hist = acc_hist + 1
                            if test_acc >= best_acc:
                                # 保存最好数据
                                best_acc = test_acc
                                # 保存最好的模型参数值状态
                                torch.save(self.audio_network.state_dict(), self.save_dir+str(cnt)+".pth")
                                cnt = (cnt+1)%5
                        if acc_hist == 24:
                            return
    def evaluate(self,i):
        self.audio_network.eval()
        correct = 0
        total = 0
        for test_input, test_label in self.test_iter:
            test_input, test_label= test_input.to(self.device), test_label.to(self.device)

            with torch.no_grad():
                out = self.audio_network(test_input)
                prediction = out.argmax(dim = 1)
            correct += torch.eq(prediction, test_label).float().sum().item()
            total += test_label.shape[0]
        self.writer.add_scalar('Accuracy/train', correct/total, self.current_iter+i*self.total_iter)
        return correct/total

    def test(self, file_path, i = 0):
        self.audio_network.load_state_dict(torch.load(self.save_dir+str(i)+".pth"))
        self.audio_network.eval()

        t_iter = MFCCDataset(data_dir=file_path)
        t_iter = DataLoader(t_iter, batch_size=1,
                                     shuffle = False)

        true_label = []
        predicted_label = []
        for t_input, t_label in t_iter:
            true_label.append(t_label.item())
            t_input = t_input.to(self.device)
            with torch.no_grad():
                out = self.audio_network(t_input).argmax(dim = 1)
                predicted_label.append(out.to('cpu').item())
        return true_label, predicted_label

    def test_single(self, path, i=0):
        padding = 50000
        waveform, sample_rate = librosa.load(path)
        wave_len = len(waveform)
        print(wave_len)
        left_pad = (padding - wave_len) // 2
        right_pad = padding - wave_len - left_pad
        print(left_pad, right_pad)
        waveform = np.pad(waveform, (left_pad, right_pad), 'wrap')
        fbank = librosa.stft(waveform, hop_length=512, n_fft=1024)
        fbank = np.abs(fbank) ** 2
        fbank = librosa.feature.melspectrogram(S=fbank, sr=sample_rate, n_mels=128)
        fbank = librosa.power_to_db(fbank, ref=np.max)
        fbank = fbank[:96, :]
        fbank = fbank[np.newaxis, :]
        fbank = fbank[np.newaxis, :]
        fbank = torch.tensor(fbank)
        fbank = fbank.to(self.device)

        self.audio_network.load_state_dict(torch.load(self.save_dir+str(i)+".pth"))
        self.audio_network.eval()

        with torch.no_grad():
            out = self.audio_network(fbank).argmax(dim = 1).to('cpu')
        return result[out[0].item()]


def plot_cm(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=range(20),
                yticklabels=range(20))
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.title('Confusion Matrix')
    plt.show()

if __name__ == "__main__":
    worker = Runner(**configs)
    worker.load_dataset(data_dir="voice")
    worker.train()
    worker.writer.close()
    true_label, predicted_label= worker.test("test_voice",2)
    plot_cm(true_label, predicted_label)
    #pred = worker.test_single(path = "upload/0000-2024-06-15-12-14-11.wav")
    #print("evaluation is:" , pred)



