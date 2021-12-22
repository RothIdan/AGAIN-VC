import numpy as np
import torch

import matplotlib.pyplot as plt
import librosa
import librosa.display
import soundfile as sf
import logging

from util.config import Config
from util.vocoder import get_vocoder

from librosa.filters import mel as librosa_mel_fn

from vocoder.inference import get_mel_AGAIN
from vocoder.meldataset import load_wav


logger = logging.getLogger(__name__)


class Dsp():
    def __init__(self, config=None):
        self.load_config(config)
        self.build_mel_basis()

        # another modules
        self.vocoder = None
        self.s3prl = None
        self.resemblyzer = None

    def load_config(self, config):
        default = {
            'n_fft': 1024,
            'hop_length': 256,
            'win_length': 1024,
            'sample_rate': 22050,
            'n_mels': 80,
            'f_min': 0,
            'f_max': 11025,
            'trim': 20,
        }
        if config is None:
            logger.info('Dsp config is None, use default config.')
            config = default
        self.config = Config(config)
        logger.info(self.config)
        for k, v in default.items():
            if k not in self.config.keys():
                self.config[k] = v

    ####################### load wav for MEL-GAN ######################
    def load_wav(self, path):
        y, sr = librosa.load(path, sr=self.config.sample_rate)
        if type(self.config.trim) is int:
            y, _ = librosa.effects.trim(y, top_db=self.config.trim)
        y = np.clip(y, -1.0, 1.0)

    ####################### load wav for HIFI-GAN ######################
    # def load_wav(self, path):
    #     y, sr = load_wav(path)
    #     y = librosa.resample(y.astype(float), sr, self.config.sample_rate)
    #     if type(self.config.trim) is int:
    #         y, _ = librosa.effects.trim(y, top_db=self.config.trim)
        
        return y, sr ###############################   sr added ################################################################################

    def save_wav(self, y, path):
        sf.write(file=path, data=y, samplerate=self.config.sample_rate)

    def build_mel_basis(self):
        self.mel_basis = librosa.filters.mel(self.config.sample_rate, self.config.n_fft,
                fmin=self.config.f_min, fmax=self.config.f_max,
                n_mels=self.config.n_mels)

####################### mel spectogram for MEL-GAN ######################
    def wav2mel(self, y, config=None):
        D = np.abs(librosa.stft(y, n_fft=self.config.n_fft,
            hop_length=self.config.hop_length, win_length=self.config.win_length)**2)
        D = np.sqrt(D)
        S = np.dot(self.mel_basis, D)
        log_S = np.log10(S)
        return log_S

####################### mel spectogram for HIFI-GAN ######################
    # def wav2mel(self, y, config):
    #     mel = get_mel_AGAIN(y, config)
    #     return mel


    # def wav2mel(self, y):
    #     print('Hi Zahi, How are you?')
    #     if torch.cuda.is_available:
    #         device = 'cuda'
    #     else:
    #         device = 'cpu'

    #     mel = librosa_mel_fn(self.config.sample_rate, self.config.n_fft, self.config.n_mels, self.config.f_min, self.config.f_max)
    #     mel_basis = torch.from_numpy(mel).float().to(device)
    #     hann_window = torch.hann_window(self.config.win_length).to(device)
    #     # y = torch.from_numpy(y)
    #     # print(f"shape is: {y.unsqueeze(1).shape}")
    #     y = np.pad(y, (int((self.config.n_fft-self.config.hop_length)/2), int((self.config.n_fft-self.config.hop_length)/2)), mode='reflect')
    #     # y = torch.nn.functional.pad(y, (int((self.config.n_fft-self.config.hop_length)/2), int((self.config.n_fft-self.config.hop_length)/2)), mode='reflect')
    #     # y = y.squeeze(1)
    #     y = torch.from_numpy(y).to(device)
    #     # print(f"shape is: {self.config.n_fft} {self.config.n_mels}")
    #     spec = torch.stft(y, self.config.n_fft, hop_length=self.config.hop_length, win_length=self.config.win_length, window=hann_window,
    #                     center=False, pad_mode='reflect', normalized=False, onesided=True)

    #     spec = torch.sqrt(spec.pow(2).sum(-1)+(1e-9))

    #     spec = torch.matmul(mel_basis, spec)
    #     spec = torch.log(torch.clamp(spec, min=1e-5) * 1)

    #     return spec.cpu().detach().numpy()

    def mel2wav(self, mel, save=''):
        if self.vocoder is None:
            self.build_vocoder()
        return self.vocoder.mel2wav(mel, save=save)

    def build_vocoder(self):
        if torch.cuda.is_available:
            device = 'cuda'
        else:
            device = 'cpu'
        self.vocoder = get_vocoder(device=device)

    def wav2s3prl_spec(self, wav):
        if self.s3prl is None:
            from s3prl import S3prl
            self.s3prl = S3prl()
        ret = self.s3prl.wav2spec(wav)
        return ret

    def wav2resemblyzer(self, wav):
        if self.resemblyzer is None:
            from resemblyzer import VoiceEncoder
            self.resemblyzer = VoiceEncoder()
        ret = self.resemblyzer.embed_utterance(wav)
        return ret

    @staticmethod
    def plot_spectrogram(mag, save=''):
        librosa.display.specshow(mag, x_axis='off', cmap='viridis')
        plt.title('spectrogram')
        if save != '':
            plt.savefig(save, format='jpg')
            plt.close()
        else:
            plt.show()
