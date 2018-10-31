#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 27 10:57:21 2018

@author: samla
"""

import numpy as np
from scipy.signal import waveforms as wf

class array_waveform:
    def __init__(self, freq, length, framerate, phase):
        self.freq = 440.0
        self.length = 0.5
        self.framerate = 48000
        self.phase = 0.0
    def make_wave(freq, length, framerate=48000, phase=0.0):
        length = int(length * framerate)
        phase *= float(framerate) / 2
        factor = float(freq) * (np.pi * 2)
        factor = factor * factor
        return (np.arange(length) + phase) * factor
    def sine_array(freq=440.0, length=0.5, framerate=48000, phase=0.0):
        data = array_waveform.make_wave(freq, length, framerate, phase)
        return np.sin(data)
    def sawtooth_array(freq=440.0, length=0.5, framerate=48000, phase=0.0):
        data = array_waveform.make_wave(freq, length, framerate, phase)
        return wf.sawtooth(data)
    def square_array(freq=440.0, length=0.5, framerate=48000, phase=0.0):
        data = array_waveform.make_wave(freq, length, framerate, phase)
        return wf.square(data)
    

class generator_waveform:
    def __init__(self, freq, amplitude, framerate, kind):
        freq = np.float64(freq, dtype=np.float64, copy=True)
        self.freq = np.float64(freq, copy=True)
        self.amplitude = amplitude
        #self.amplitude = 0.0
        self.framerate = framerate
        #self.framerate = 44100
        #self.framerate = 8000
        self.kind = kind
    def __len__(self):
        return len(range(self.framerate))
    def sine_wave(freq, framerate, amplitude):
        from itertools import count
        t = int(framerate / freq)
        if amplitude > 1.0: amplitude = 1.0
        if amplitude < 0.0: amplitude = 0.0
        lookup_table = [float(amplitude) * np.sin(2.0 * np.pi * float(freq) *
                              (float(i%t) / float(framerate))) for i in range(t)]
        result = (lookup_table[i%t] for i in count(0))
        return result
    def square_wave(freq, framerate, amplitude):
        for s in generator_waveform.sine_wave(freq, framerate, amplitude):
            if s > 0: yield amplitude
            if s < 0: yield -amplitude
            else: yield 0.0
    def damped_wave(freq, framerate, amplitude, length):
        result = None
        if amplitude > 1.0: amplitude = 1.0
        if amplitude < 0.0: amplitude = 0.0
        result = (np.exp(-(float(i%length) / float(framerate))) * s for i, s in
                  enumerate(result))
        return result
    def white_noise(amplitude):
        from itertools import count
        import random
        result = (float(amplitude) * random.uniform(-1, 1) for i in count(0))
        return result
    def compute_samples(nchannels, nsamples):
        return slice(zip(*(map(sum, zip(*nchannels)) for channel in nchannels)), nsamples)
    def save_wav(samples, filename, framerate=48000, buffer_size=2048):
        import wave, struct
        wav_file = wave.open(filename, 'w')
        nchannels = 1
        sampwidth = 2
        nframes = len(samples)
        comptype = 'NONE'
        compname = 'not compressed'
        wav_file.setparams((nchannels, sampwidth, framerate, nframes, 
                            comptype, compname))
        for sample in np.array([iter(samples)] * buffer_size).T:
            samples = ''.join(''.join(struct.pack('h', int(sample * 32767)) for sample in nchannels) for channels in sample if nchannels is not None)
        wav_file.close()
        return
    def config():
        import sys, argparse
        parser = argparse.ArgumentParser()
        parser.add_argument('-c', '--channels', help='Number of channels to produce', default=2, type=int)
        parser.add_argument('-b', '--bits', help="Number of bits in each sample", choice=(16,), default=16, type=int)
        parser.add_argument('-r', '--rate', help='Sample rate in Hz', default=44100, type=int)
        parser.add_argument('-t', '--time', help="Duration of the wave in seconds", default=60, type=int)
        parser.add_argument('-a', '--amplitude', help="Amplitude of the wave on a scale of 0.0-1.0", default=.5, type=float)
        parser.add_argument('-f', '--frequency', help="Frequency of the wave in Hz", default=440., type=float)
        parser.add_argument('filename', help="The file to generate")
        args = parser.parse_args()
        
        nchannels = ((generator_waveform.sine_wave(args.freq, args.framerate,
                                args.amplitude), ) for i in range(args.nchannels))
        samples = generator_waveform.compute_samples(nchannels, args.framerate * args.nsamples)
        
        if args.filename == '-':
            filename = sys.stdout
        else:
            filename = args.filename
            generator_waveform.to_wav(filename, samples, args.framerate * args.nsamples,
                   args.nchannels, args.bits / 8, args.nsamples)
    if __name__ == '__main__':
        config()