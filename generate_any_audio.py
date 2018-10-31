# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 13:30:32 2018

@author: samla
"""

import struct
import numpy as np
from scipy import signal as sg

fs = 44100
f = 440
sample = 44100

x = np.arange(sample)

sine = 100 * np.sin(2 * np.pi * f * x/fs)
square = 100 * sg.square(2 * np.pi * f * x/fs)
square_duty = 100 * sg.square(2 * np.pi * f * x/fs, duty = .8)
sawtooth = 100 * sg.sawtooth(2 * np.pi * f * x/fs)

'''
def save_wav(file_name):
    wav_file = wave.open(file_name, 'w')
    
    nchannels = 1
    sampwidth = 2
    
    nframes = len(audio)
    comptype = 'NONE'
    compname = 'not compressed'
    
    wav_file.setparams((nchannels, sampwidth, sample_rate, nframes, comptype, compname))
    
    for sample in audio:
        wav_file.writeframes(struct.pack('h', int(sample * 32767.0)))
        
    wav_file.close()
    
    return
'''
