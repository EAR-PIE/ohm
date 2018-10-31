# -*- coding: utf-8 -*-
"""
Created on Mon Oct  1 21:58:06 2018

@author: samla
"""

import numpy as np
import wave, struct

frame_rate = 44100
audio = []

def generate_sine_wave(freq, duration, vol):
    global audio
    nsamples = duration * (frame_rate / 1000.0)
    for x in range(int(nsamples)):
        audio.append(vol * np.sin(2 * np.pi * freq * (x / frame_rate)))
        
    return(audio)

def write_wav_file(file_name):
    wav = wave.open(file_name, 'w')
    nchannels = 1
    sampwidth = 2
    nframes = len(audio)
    comptype = 'NONE'
    compname = 'not compressed'
    wav.setparams((nchannels, sampwidth, frame_rate, nframes, comptype, compname))
    for sample in audio:
        wav.writeframes(struct.pack('h', int(sample * 32767.0)))
    wav.close()
    return

def main():
    pass

if __name__ == '__main()__':
    main()