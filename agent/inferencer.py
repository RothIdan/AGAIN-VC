import os
import torch
import logging
import numpy as np
from glob import glob
from tqdm import tqdm

import librosa
import crepe
from scipy.io import wavfile
from vocoder.inference_e2e import hifi_gan_mel2wav


# from vocoder.inference import get_mel_AGAIN ##################################
# from vocoder.meldataset import load_wav ################################

from preprocessor.base import preprocess_one
from .base import BaseAgent
from util.dsp import Dsp



logger = logging.getLogger(__name__)

def gen_wav_list(path):
    if os.path.isdir(path):
        wav_list = glob(os.path.join(path, '*.wav'))
    elif os.path.isfile(path):
        wav_list = [path]
    else:
        raise NotImplementedError(f'{path} is invalid for generating wave file list.')
    return wav_list

class WaveData():
    def __init__(self, path):
        self.path = path
        self.processed = False
        self.data = {}

    def set_processed(self):
        self.processed = True

    def is_processed(self):
        return self.processed
    
    def __getitem__(self, key):
        if type(key) is str:
            return self.data[key]
        else:
            raise NotImplementedError
    
    def __setitem__(self, key, value):
        if type(key) is str:
            self.data[key] = value
        else:
            raise NotImplementedError


########################################################
class spec:
        def __init__(self, s):
            self.data = {}
            self.data['mel'] = s


############################################################



class Inferencer(BaseAgent):
    def __init__(self, config, args):
        super().__init__(config, args)
        self.indexes_path = config.dataset.indexes_path
        self.dsp_modules = {}
        for feat in config.dataset.feat:
            if feat in self.dsp_modules.keys():
                module = self.dsp_modules[feat]
            else:
                module = Dsp(args.dsp_config.feat[feat])
                self.dsp_modules[feat] = module
        self.model_state, self.step_fn = self.build_model(config.build)
        self.model_state = self.load_model(self.model_state, args.load, device=self.device)
        ################################################################################################
        self.config = config

    def build_model(self, build_config):
        return super().build_model(build_config, mode='inference', device=self.device)

    def load_wav_data(self, source_path, target_path, out_path):
        # load wavefiles
        sources = gen_wav_list(source_path)
        assert(len(sources) > 0), f'Source path "{source_path}"" should be a wavefile or a directory which contains wavefiles.'
        targets = gen_wav_list(target_path)
        assert(len(targets) > 0), f'Target path "{target_path}" should be a wavefile or a directory which contains wavefiles.'
        if os.path.exists(out_path):
            assert(os.path.isdir(out_path)), f'Output path "{out_path}" should be a directory.'
        else:
            os.makedirs(out_path)
            logger.info(f'Output directory "{out_path}" is created.')
        os.makedirs(out_path, exist_ok=True)
        os.makedirs(os.path.join(out_path, 'wav_mel_gan'), exist_ok=True)
        os.makedirs(os.path.join(out_path, 'plt'), exist_ok=True)
        os.makedirs(os.path.join(out_path, 'mel'), exist_ok=True)
        os.makedirs(os.path.join(out_path, 'npy'), exist_ok=True)

        for i, source in enumerate(sources):
            sources[i] = WaveData(source)
        for i, target in enumerate(targets):
            targets[i] = WaveData(target)

        return sources, targets, out_path

    def process_wave_data(self, wav_data, seglen=None):
        if wav_data.is_processed():
            return
        else:
            wav_path = wav_data.path
            basename = os.path.basename(wav_path)
            for feat, module in self.dsp_modules.items():
                wav_data[feat] = preprocess_one((wav_path, basename), module, config=self.config) ##############################
                if seglen is not None:
                    wav_data[feat] = wav_data[feat][:,:seglen]
                wav_data.set_processed()
            return
    
    ###################################################################################################################################
    def load_wav(self, path, dsp_module):
        y, sr = librosa.load(path, sr=dsp_module.config.sample_rate)
        if type(dsp_module.config.trim) is int:
            y, _ = librosa.effects.trim(y, top_db=dsp_module.config.trim)
        y = np.clip(y, -1.0, 1.0)

        return y, sr
        
    ###################################################################################################################################



    # ====================================================
    #  inference
    # ====================================================
    def inference(self, source_path, target_path, out_path, seglen):
        # sources and targets contain 1 source and 1 target vars when running inference with 2 wav files 
        sources, targets, out_path = self.load_wav_data(source_path, target_path, out_path) 
        with torch.no_grad():
            # source and target are of type class WaveData: has attribute 'data' which is a dictionary with 1 key: 'mel'
            # source.data['mel'] contains a np array of the mel spectogram of size (80,~)
            # we think that 80 represnts c_in = 80 in the config file
            for i, source in enumerate(sources):
                logger.info(f'Source: {source.path}')
                for j, target in enumerate(targets):
                    logger.info(f'Target: {target.path}')
                    source_basename = os.path.basename(source.path).split('.wav')[0]
                    target_basename = os.path.basename(target.path).split('.wav')[0]
                    output_basename = f'{source_basename}_to_{target_basename}'
                    output_wav = os.path.join(out_path, 'wav_mel_gan', output_basename+'.wav')
                    output_plt = os.path.join(out_path, 'plt', output_basename+'.png')

                    source_path = source.path
                    target_path = target.path
                    # The following two line is for generating the mel-spectogram using melgan.
                    self.process_wave_data(source, seglen=seglen)
                    # self.dsp.mel2wav(source['melgan'], os.path.join('data/tmp/mos_melgan/', source_basename+'.wav'))
                    # continue
                    self.process_wave_data(target, seglen=seglen)
##################################################################################################################################

                    # hop_length = self.dsp_modules['mel'].config.hop_length
                    # # load and preprocess the wav file 
                    # # self.dsp_modules['mel'] is a DSP object with config method that have all the preprocess.yaml info
                    # src_audio, sr = self.load_wav(source_path, self.dsp_modules['mel'])
                    # tar_audio, sr = self.load_wav(target_path, self.dsp_modules['mel'])
                    # # frequency is a np array which contains the pitch frequency in Hz
                    # # step size is 10ms by default
                    # _, src_freq, _, _ = crepe.predict(src_audio, sr, viterbi=True, step_size=1000*(hop_length/sr))
                    # _, tar_freq, _, _ = crepe.predict(tar_audio, sr, viterbi=True, step_size=1000*(hop_length/sr))
                    
                    # # reshaping if needed
                    # src_mel_len = source.data['mel'].shape[1]
                    # tar_mel_len = target.data['mel'].shape[1]
                    # if len(src_freq) < src_mel_len:
                    #     pad_sz = src_mel_len-len(src_freq)
                    #     src_frec = np.pad(src_frec, (0,pad_sz), mode='constant')
                    # if len(tar_freq) < tar_mel_len:
                    #     pad_sz = tar_mel_len-len(tar_freq)
                    #     tar_freq = np.pad(tar_freq, (0,pad_sz), mode='constant')

                    # # Concatenating the pitch estimation to the input Mel-spectograms
                    # source.data['mel'] = np.concatenate((source.data['mel'], [src_freq[:src_mel_len]]))
                    # target.data['mel'] = np.concatenate((target.data['mel'], [tar_freq[:tar_mel_len]]))


#####################################################################################################################################

                    # print(f'************{source.data["mel"].shape}***************')
                    # print(f"mel:{source.data['mel'].shape}")
                    data = {
                        'source': source,
                        'target': target,
                    }
                    meta = self.step_fn(self.model_state, data)
                    # dec containes the converted mel-spectogram generated after the decoder
                    dec = meta['dec']
                    # wav file synthesize using Mel-GAN
                    generated_wav = self.mel2wav(dec, output_wav)
                    Dsp.plot_spectrogram(dec.squeeze().cpu().numpy(), output_plt)


##########################################################################################################################
                    # MAX_WAV_VALUE = 32768.0
                    # # s_wav, sr = load_wav(source_path)
                    # # s_wav = librosa.resample(s_wav.astype(float), sr, 22050)
                    # # s_wav = s_wav / MAX_WAV_VALUE
                    # # s_wav = torch.FloatTensor(s_wav).to(self.device)
                    # # s_spec = get_mel_AGAIN(s_wav.unsqueeze(0)).squeeze(0).cpu().numpy()
                    # # source_spec = spec(s_spec)
                    # # source_spec = {'mel':s_spec}
                    # # print(f"type hifi:{type(source_spec)},  shape: {source_spec.shape}")

                    # # t_wav, sr = load_wav(target_path)
                    # # t_wav = librosa.resample(t_wav.astype(float), sr, 22050)
                    # # t_wav = t_wav / MAX_WAV_VALUE
                    # # t_wav = torch.FloatTensor(t_wav).to(self.device)
                    # # t_spec = get_mel_AGAIN(t_wav.unsqueeze(0)).squeeze(0).cpu().numpy()
                    # # target_spec = spec(t_spec)
                    # # target_spec = {'mel':t_spec}

                    # hop_length = self.dsp_modules['mel'].config.hop_length
                    # # load and preprocess the wav file 
                    # # self.dsp_modules['mel'] is a DSP object with config method that have all the preprocess.yaml info
                    # src_audio, sr = load_wav(source_path)
                    # src_audio = librosa.resample(src_audio.astype(float), sr, 22050)
                    # src_audio = src_audio / MAX_WAV_VALUE
                    # # src_audio, sr = self.load_wav(source_path, self.dsp_modules['mel'])
                    # _, src_freq, _, _ = crepe.predict(src_audio, sr, viterbi=True, step_size=1000*(hop_length/sr))
                    # src_audio = torch.FloatTensor(src_audio).to(self.device)
                    # s_spec = get_mel_AGAIN(src_audio.unsqueeze(0)).squeeze(0).cpu().numpy()
                    # source_spec = spec(s_spec)
                    # source_spec = {'mel':s_spec}

                    # # tar_audio, sr = self.load_wav(target_path, self.dsp_modules['mel'])
                    # tar_audio, sr = load_wav(target_path)
                    # tar_audio = librosa.resample(tar_audio.astype(float), sr, 22050)
                    # tar_audio = tar_audio / MAX_WAV_VALUE
                    # _, tar_freq, _, _ = crepe.predict(tar_audio, sr, viterbi=True, step_size=1000*(hop_length/sr))
                    # tar_audio = torch.FloatTensor(tar_audio).to(self.device)
                    # t_spec = get_mel_AGAIN(tar_audio.unsqueeze(0)).squeeze(0).cpu().numpy()
                    # target_spec = spec(t_spec)
                    # target_spec = {'mel':t_spec}                   
                    
                    # # # reshaping if needed
                    # # src_mel_len = source.data['mel'].shape[1]
                    # # tar_mel_len = target.data['mel'].shape[1]
                    # # if len(src_freq) < src_mel_len:
                    # #     pad_sz = src_mel_len-len(src_freq)
                    # #     src_frec = np.pad(src_frec, (0,pad_sz), mode='constant')
                    # # if len(tar_freq) < tar_mel_len:
                    # #     pad_sz = tar_mel_len-len(tar_freq)
                    # #     tar_freq = np.pad(tar_freq, (0,pad_sz), mode='constant')

                    # # Concatenating the pitch estimation to the input Mel-spectograms
                    # source_spec['mel'] = np.concatenate((source_spec['mel'], [src_freq[:source_spec['mel'].shape[1]]]))
                    # target_spec['mel'] = np.concatenate((target_spec['mel'], [tar_freq[:target_spec['mel'].shape[1]]]))


                    # data = {
                    #     'source': source_spec,
                    #     'target': target_spec,
                    # }
                    # print(f"type hifi:{data['source']['mel'].shape}")
                    # meta = self.step_fn(self.model_state, data)
                    # # dec containes the converted mel-spectogram generated after the decoder
                    # dec = meta['dec']






                    # dec = dec.squeeze()
                    # dec = dec[None].to(self.device).float()
                    
                    # wav file synthesize using HiFi-GAN and saved at file
                    hifi_gan_mel2wav(dec, self.device, output_basename)  # HiFi-GAN Vocoder
                    
##########################################################################################################################

                    source_plt = os.path.join(out_path, 'plt', f'{source_basename}.png')
                    Dsp.plot_spectrogram(source['mel'], source_plt)
                    np.save(os.path.join(out_path, 'mel', f'{source_basename}.npy'), source['mel'])

                    # np.save(os.path.join(out_path, 'npy', f'{source_basename}.npy'), dec.cpu())
        logger.info(f'The generated files are saved to {out_path}.')