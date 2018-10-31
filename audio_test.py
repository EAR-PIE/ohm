# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 11:45:23 2018

@author: samla
"""

import audioread, simpleaudio, io
import scipy.io.wavfile as wavfile
import numpy as np

def get_file(filename):
    filename = str(input('Enter File Name or Path: '))
    file_data = audioread.audio_open(filename)
    data = [x for x in file_data] # generate list of bytestrings
    data = b''.join(data) # join bytestrings into a single urbytestring
    numeric_data = np.fromstring(data, '<i2') # convert from audioread object to numeric
    sample_rate = file_data.samplerate
    return numeric_data, sample_rate

def write_file(numeric_data, sample_rate):
    # generate a wave file in memory
    memory_file = io.BytesIO() # set buffer to write to
    wavfile.write(memory_file, sample_rate, numeric_data)
    return memory_file

def play_file(memory_file):
    wave_obj = simpleaudio.WaveObject.from_wave_file(memory_file)
    play_obj = wave_obj.play()
    play_obj.wait_done()
    return