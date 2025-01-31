import os
import logging
import numpy as np
from tqdm import tqdm
from functools import partial
from multiprocessing.pool import ThreadPool

from util.dsp import Dsp

import crepe
import torch

logger = logging.getLogger(__name__)

def preprocess_one(input_items, module, output_path='', config=None):
    input_path, basename = input_items
    # load and preprocess the wav file and it's sample rate
    y_old, sr = module.load_wav(input_path)
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    MAX_WAV_VALUE = 32768.0
    y = y_old / MAX_WAV_VALUE
    y = torch.FloatTensor(y).to(device)
    # x = get_mel(wav.unsqueeze(0))

    if module.config.dtype == 'wav':
        ret = y_old

    ##########################################################################################################################
    # elif module.config.dtype == 'melspectrogram': 
    #     ret = module.wav2mel(y)

    
    # elif module.config.dtype == 'mel_crepe':


    elif module.config.dtype == 'melspectrogram':
        if config == None or config.dataset.in_type == 'spectogram_pitch':  
            mel = module.wav2mel(y_old) ######################## use for MEL-GAN ############################

            ##################### use for HIFI-GAN ######################
            # mel = module.wav2mel(y.unsqueeze(0), module)
            # mel = mel.squeeze(0).cpu().numpy()

            hop_length = module.config.hop_length
            # pitch estimation - frequency is a np array which contains the pitch frequency in Hz
            # step size is 10ms by default
            _, freq, _, _ = crepe.predict(y_old, sr, viterbi=True, step_size=1000*(hop_length/sr))
            freq = np.sin(freq) ############################################################################################
            # reshaping if needed
            mel_len = mel.shape[1]
            if len(freq) < mel_len:
                pad_sz = mel_len - len(freq)
                freq = np.pad(freq, (0,pad_sz), mode='constant')
            # Concatenating the pitch estimation to the Mel-spectograms
            ret = np.concatenate((mel, [freq[:mel_len]]))
        elif config.dataset.in_type == 'spectogram':
            ret = module.wav2mel(y_old, module)


    
    ###########################################################################################################################
    elif module.config.dtype == 'f0':
        f0, sp, ap = pw.wav2world(y.astype(np.float64), module.config.sample_rate)
        ret = f0
        if (f0 == 0).all():
            logger.warn(f'f0 returns all zeros: {input_path}')
    elif module.config.dtype == 's3prl_spec':
        ret = module.wav2s3prl_spec(y)
        if ret is None:
            logger.warn(f'S3PRL spectrogram returns NoneType: {input_path}')
    elif module.config.dtype == 'resemblyzer':
        y = resemblyzer.preprocess_wav(input_path)
        ret = module.wav2resemblyzer(y)
    else:
        logger.warn(f'Not implement feature type {module.config.dtype}')
    if output_path == '':
        return ret
    else:
        if type(ret) is np.ndarray:
            np.save(os.path.join(output_path, f'{basename}.npy'), ret)
        else:
            logger.warn(f'Feature {module.config.dtype} is not saved: {input_path}.')
        return 1


class BasePreproceccor():
    def __init__(self, config):
        self.dsp_modules = {}
        for feat in config.feat_to_preprocess:
            self.dsp_modules[feat] = Dsp(config.feat[feat])

    def preprocess(self, input_path, output_path, feat, njobs):
        file_dict = self.gen_file_dict(input_path)
        logger.info(f'Starting to preprocess from {input_path}.')
        self.preprocess_from_file_dict(file_dict=file_dict, output_path=output_path, feat=feat, njobs=njobs)
        logger.info(f'Saving processed file to {output_path}.')
        return

    def preprocess_from_file_dict(self, file_dict, output_path, feat, njobs):
        os.makedirs(os.path.join(output_path, feat), exist_ok=True)
        module = self.dsp_modules[feat]
        task = partial(preprocess_one, module=module, output_path=os.path.join(output_path, feat))
        with ThreadPool(njobs) as pool:
            _ = list(tqdm(pool.imap(task, file_dict.items()), total=len(file_dict), desc=f'Preprocessing '))

    def gen_file_dict(self, input_path):
        raise NotImplementedError
