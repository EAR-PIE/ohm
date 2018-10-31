# -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 23:27:30 2018

@author: samla
"""
import math, wave, struct

audio = []
sample_rate = 44100.0

def append_silence(duration_ms=500):
    num_samples = duration_ms * (sample_rate / 1000.0)
    
    for x in range(int(num_samples)):
        audio.append(0.0)
        
    return

def append_sine(freq, duration_ms, vol):
    global audio
    
    num_samples = duration_ms * (sample_rate / 1000.0)
    
    for x in range(int(num_samples)):
        audio.append(vol * math.sin(2 * math.pi * freq * (x / sample_rate)))
        
    return

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

def main():
    pass
if __name__ == '__main()__':
    main()

'''
append_sine(440., 500, .5)
append_silence()
append_sine(260., 500, .5)
append_silence()
save_wav('output.wav')'''