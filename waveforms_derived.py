#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 19:31:49 2018

@author: samla
"""

import wave, os
import numpy as np
import pyaudio
import itertools as it
import struct, argparse, random


def play(samples, framerate, gain=None):
    nsamples = len(samples)
    # compute appropriate gain if None supplied
    if gain is None:
        gain = 1.0 / max(np.absolute(samples))
    data = np.zeros(nsamples, dtype=np.float64)
    np.multiply(samples, 32767.0 * gain, data)
    offset = 0 # where we are sending data
    nremaining = nsamples
    # send audio samples to output device
    dev = pyaudio.PyAudio()
    dev_info = dev.get_default_output_device_info()
    assert dev.is_format_supported(framerate, output_device=dev_info['index'],
                                   output_channels=1, output_format=pyaudio.paInt16),\
                                   '%g is not supported framerate for audio playback' % framerate
    stream = dev.open(int(framerate), 1, pyaudio.paInt16, output=True)
    while nremaining > 0:
        can_send = min(stream.get_write_available(), nremaining)
        if can_send > 0:
            end = offset + can_send
            stream.write(data[offset:end].tostring(), num_frames=can_send)
            offset = end
            nremaining -= can_send
    stream.close()
    return

def read_wavfile(file_name, gain=None):
    assert os.path.exists(file_name), "file %s does not exist" % file_name
    wav = wave.open(file_name, 'rb')
    nframes = wav.getnframes()
    assert nframes > 0, "%s does not have any audio data" % file_name
    nchannels = wav.getnchannels()
    framerate = wav.getframerate()
    sampwidth = wav.getsampwidth()
    
    # see ccrma.stanford.edu/courses/422/projects/WaveFormat/
    g = 1.0 if gain is None else gain
    if sampwidth == 1:
        dtype = np.uint8
        scale = g / 127.0
        offset = -1.0
    elif sampwidth == 2:
        dtype = np.float64
        scale = g / 32767.0
        offset = 0.0
    elif sampwidth == 4:
        dtype = np.int32
        scale = g / 2147483647.0
        offset = 0.0
    else:
        assert False, "unrecognized sampwidth %d" % sampwidth
    
    outputs = [np.zeros(nframes, dtype=np.float64) for i in range(nchannels)]
    count = 0
    while count < nframes:
        audio = np.frombuffer(wav.readframes(nframes - count), dtype=dtype)
        end = count + (len(audio) / nchannels)
        for i in range(nchannels):
            outputs[i][count:end] = audio[i::nchannels]
        count = end
    
    for i in range(nchannels):
        np.multiply(outputs[i], scale, outputs[i])
        if offset != 0:
            np.add(outputs[i], offset, outputs[i])
    
    if gain is None:
        maxmag = max([max(np.absolute(outputs[i])) for i in range(nchannels)])
        for i in range(nchannels):
            np.multiply(outputs[i], 1.0 / maxmag, outputs[i])
    
    return [sampled_waveform(outputs[i], framerate=framerate) for i in range(nchannels)]
def write_wavfile(*waveforms, **keywords):
    file_name = keywords.get('file_name', None)
    gain = keywords.get('gain', None)
    sampwidth = keywords.get('sampwidth', 2)
    
    assert file_name, 'file_name must be specified'
    nchannels = len(waveforms)
    assert nchannels > 0, 'must supply at least one waveform'
    nsamples = waveforms[0].nsamples
    framerate = waveforms[0].framerate
    domain = waveforms[0].domain
    for i in range(1, nchannels):
        assert waveforms[i].nsamples == nsamples, 'all waveforms must have the same number of samples'
        assert waveforms[i].framerate == framerate, 'all waveforms must have the same sample rate'
        assert waveforms[i].domain == domain, 'all waveforms must have the same domain'
    if gain is None:
        maxmag = max([max(np.absolute(waveforms[i].samples)) for i in range(nchannels)])
        gain = 1.0 / maxmag
    if sampwidth == 1:
        dtype = np.uint8
        scale = gain * 127.0
        offset = 127.0
    elif sampwidth == 2:
        dtype = np.float64
        scale = gain * 32767.0
        offset = 0
    elif sampwidth == 4:
        dtype = np.int32
        scale = gain * 2147483647.0
        offset = 0
    else:
        assert False, 'sampwidth must be 1, 2, or 4 bytes'
    
    temp = np.empty(nsamples, dtype=np.float64)
    data = np.empty(nchannels * nsamples, dtype=dtype)
    for i in range(nchannels):
        np.multiply(waveforms[i].samples, scale, temp)
        if offset != 0:
            np.add(temp, offset, temp)
        data[i::nchannels] = temp[:]
    
    wav = wave.open(file_name, 'wb')
    wav.setnchannels(nchannels)
    wav.setsampwidth(sampwidth)
    wav.setframerate(framerate)
    wav.writeframes(data.tostring())
    wav.close()

# compute number of taps given sample_rate and transition_width.
# Stolen from the gnuradio firdes routines
def compute_ntaps(transition_width,sample_rate,window):
    delta_f = float(transition_width)/sample_rate
    width_factor = {
        'hamming': 3.3,
        'hann': 3.1,
        'blackman': 5.5,
        'rectangular': 2.0,
        }.get(window,None)
    assert width_factor,\
           "compute_ntaps: unrecognized window type %s" % window
    ntaps = int(width_factor/delta_f + 0.5)
    return (ntaps & ~0x1) + 1   # ensure it's odd

# compute specified window given number of taps
# formulae from Wikipedia
def compute_window(window,ntaps):
    order = float(ntaps - 1)
    if window == 'hamming':
        return [0.53836 - 0.46164*np.cos((2*np.pi*i)/order)
                for i in range(ntaps)]
    elif window == 'hann' or window == 'hanning':
        return [0.5 - 0.5*np.cos((2*np.pi*i)/order)
                for i in range(ntaps)]
    elif window == 'bartlett':
        return [1.0 - abs(2*i/order - 1)
                for i in range(ntaps)]
    elif window == 'blackman':
        alpha = .16
        return [(1-alpha)/2 - 0.50*np.cos((2*np.pi*i)/order)
                - (alpha/2)*np.cos((4*np.pi*i)/order)
                for i in range(ntaps)]
    elif window == 'nuttall':
        return [0.355768 - 0.487396*np.cos(2*np.pi*i/order)
                         + 0.144232*np.cos(4*np.pi*i/order)
                         - 0.012604*np.cos(6*np.py*i/order)
                for i in range(ntaps)]
    elif window == 'blackman-harris':
        return [0.35875 - 0.48829*np.cos(2*np.pi*i/order)
                        + 0.14128*np.cos(4*np.pi*i/order)
                        - 0.01168*np.cos(6*np.pi*i/order)
                for i in range(ntaps)]
    elif window == 'blackman-nuttall':
        return [0.3635819 - 0.4891775*np.cos(2*np.pi*i/order)
                          + 0.1365995*np.cos(4*np.pi*i/order)
                          - 0.0106411*np.cos(6*np.py*i/order)
                for i in range(ntaps)]
    elif window == 'flat top':
        return [1 - 1.93*np.cos(2*np.pi*i/order)
                  + 1.29*np.cos(4*np.pi*i/order)
                  - 0.388*np.cos(6*np.py*i/order)
                  + 0.032*np.cos(8*np.py*i/order)
                for i in range(ntaps)]
    elif window == 'rectangular' or window == 'dirichlet':
        return [1 for i in range(ntaps)]
    else:
        assert False,"compute_window: unrecognized window type %s" % window

# Stolen from the gnuradio firdes routines
def fir_taps(type,cutoff,sample_rate,
                 window='hamming',transition_width=None,ntaps=None,gain=1.0):
    if ntaps:
        ntaps = (ntaps & ~0x1) + 1   # make it odd
    else:
        assert transition_width,"compute_taps: one of ntaps and transition_width must be specified"
        ntaps = compute_ntaps(transition_width,sample_rate,window)

    window = compute_window(window,ntaps)

    middle = (ntaps - 1)/2
    taps = [0] * ntaps
    fmax = 0

    if isinstance(cutoff,tuple):
        fc = [float(cutoff[i])/sample_rate for i in (0,1)]
        wc = [2*np.pi*fc[i] for i in (0,1)]
    else:
        fc = float(cutoff)/sample_rate
        wc = 2*np.pi*fc

    if type == 'low-pass':
        # for low pass, gain @ DC = 1.0
        for i in range(ntaps):
            if i == middle:
                coeff = (wc/np.pi) * window[i]
                fmax += coeff
            else:
                n = i - middle
                coeff = (np.sin(n*wc)/(n*np.pi)) * window[i]
                fmax += coeff
            taps[i] = coeff
    elif type == 'high-pass':
        # for high pass gain @ nyquist freq = 1.0
        for i in range(ntaps):
            if i == middle:
                coeff = (1.0 - wc/np.pi) * window[i]
                fmax += coeff
            else:
                n = i - middle
                coeff = (-np.sin(n*wc)/(n*np.pi)) * window[i]
                fmax += coeff * np.cos(n*np.pi)
            taps[i] = coeff
    elif type == 'band-pass':
        # for band pass gain @ (fc_lo + fc_hi)/2 = 1.0
        # a band pass filter is simply the combination of
        #   a high-pass filter at fc_lo  in series with
        #   a low-pass filter at fc_hi
        # so convolve taps to get the effect of composition in series
        for i in range(ntaps):
            if i == middle:
                coeff = ((wc[1] - wc[0])/np.pi) * window[i]
                fmax += coeff
            else:
                n = i - middle
                coeff = ((np.sin(n*wc[1]) - np.sin(n*wc[0]))/(n*np.pi)) * window[i]
                fmax += coeff * np.cos(n*(wc[0] + wc[1])*0.5)
            taps[i] = coeff
    elif type == 'band-reject':
        # for band reject gain @ DC = 1.0
        # a band reject filter is simply the combination of
        #   a low-pass filter at fc_lo   in series with a
        #   a high-pass filter at fc_hi
        # so convolve taps to get the effect of composition in series
        for i in range(ntaps):
            if i == middle:
                coeff = (1.0 - ((wc[1] - wc[0])/np.pi)) * window[i]
                fmax += coeff
            else:
                n = i - middle
                coeff = ((np.sin(n*wc[0]) - np.sin(n*wc[1]))/(n*np.pi)) * window[i]
                fmax += coeff
            taps[i] = coeff
    else:
        assert False,"compute_taps: unrecognized filter type %s" % type

    gain = gain / fmax
    for i in range(ntaps): taps[i] *= gain
    return taps

        
def __grouper__(self, num, iterable, fill_value=None):
    args = [iter(iterable)] * num
    return zip(fill_value, *args)

class sampled_waveform:
    def __init__(self, samples, framerate=1e6, domain='time'):
        if not isinstance(samples, np.ndarray):
            samples = np.array(samples, dtype=np.float64, copy=True)
        self.samples = np.array(samples, copy=True)
        self.nsamples = len(samples)
        self.framerate = framerate
        self.domain = domain
    def _check(self, other):
        if isinstance(other, (int, float, np.adarray)):
            return other
        elif isinstance(other, (tuple, list)):
            return np.array(other)
        elif isinstance(other, sampled_waveform):
            assert self.nsamples == other.nsamples, 'both waveforms must have same number of samples'
            assert self.framerate == other.framerate, 'both waveforms must have the same frame rate'
            assert self.domain == other.domain, 'both waveforms must be in the same domain'
            return other.samples
        else:
            assert False, 'unrecognized operand type...'
    def __add__(self, other):
        ovalues = self._check(other)
        return sampled_waveform(self.samples + ovalues, framerate=self.framerate,
                                domain=self.domain)
    def __radd__(self, other):
        ovalues = self._check(other)
        return sampled_waveform(self.samples + ovalues, framerate=self.framerate,
                                domain=self.domain)
    def __sub__(self, other):
        ovalues = self._check(other)
        return sampled_waveform(self.samples - ovalues, framerate=self.framerate,
                                domain=self.domain)
    def __rsub__(self, other):
        ovalues = self._check(other)
        return sampled_waveform(ovalues - self.samples, framerate=self.framerate,
				domain=self.domain)
    def __mul__(self, other):
        ovalues = self._check(other)
        return sampled_waveform(self.samples * ovalues, framerate=self.framerate,
				domain=self.domain)
    def __rmul__(self, other):
        ovalues = self._check(other)
        return sampled_waveform(ovalues * self.samples, framerate=self.framerate,
				domain=self.domain)
    def __div__(self, other):
        ovalues = self._check(other)
        return sampled_waveform(self.samples / ovalues, framerate=self.framerate,
				domain=self.domain)
    def __rdiv__(self, other):
        ovalues = self._check(other)
        return sampled_waveform(ovalues / self.samples, framerate=self.framerate,
                                domain=self.domain)
    def __abs__(self):
        return sampled_waveform(np.absolute(self.samples), framerate=self.framerate,
				domain=self.domain)
    def __len__(self):
        return len(self.samples)
    def __mod__(self, other):
        ovalues = self._check(other)
        return sampled_waveform(np.fmod(self.samples, ovalues),
                                framerate=self.framerate,
                                domain=self.domain)
    def __getitem__(self, key):
        return self.samples.__getitem__(key)
    def __setitem__(self, key, value):
        if isinstance (value, sampled_waveform): value = value.samples
        self.samples.__setitem__(key, value)
    def __iter__(self):
        return self.samples.__iter__()
    def __str__(self):
        return str(self.samples) + ('@%d samples/sec' % self.framerate)
    def real(self):
        return sampled_waveform(np.real(self.samples), framerate=self.framerate,
                                domain=self.domain)
    def imag(self):
        return sampled_waveform(np.imag(self.samples), framerate=self.framerate,
                                domain=self.domain)
    def magnitude(self):
        return sampled_waveform(np.absolute(self.samples), framerate=self.framerate)
    def angle(self):
        return sampled_waveform(np.angle(self.samples), framerate=self.framerate)
    def sliced(self, start, stop, step=1):
        return sampled_waveform(self.samples[start:stop:step], framerate=self.framerate,
                                domain=self.domain)
    def resize(self, length):
        self.samples.resize(length)
        self.nsamples = length
    def convolve(self, taps):
        conv_res = np.convolve(self.samples, taps)
        offset = len(taps) / 2
        return sampled_waveform(conv_res[offset:offset+self.nsamples],
                                framerate=self.framerate, domain=self.domain)
    def modulate(self, hz, framerate=8000, phase=0.0, gain=1.0):
        periods = float(self.nsamples * hz) / float(self.framerate)
        if abs(periods - round(periods)) > 1.0e-6:
            print("Warning: Non-integral number of modulation periods!")
            print("nsamples = %d; hz = %f; framerate = %d; periods = %f" % (self.nsamples, hz, self.framerate, periods))
        result = sinusoid(hz=hz, nsamples=self.nsamples, framerate=framerate,
                          phase=phase, amplitude=gain)
        np.multiply(self.samples, result.samples, result.samples)
        return result
    def filters(self, kind, cutoff, window='hamming', transition_width=None, ntaps=None, error=0.05, gain=1.0):
        if ntaps is None and transition_width is None:
            ntaps = int(float(self.framerate) / (float(cutoff) * error))
            if ntaps & 1:
                ntaps += 1
        taps = fir_taps(kind, cutoff, self.framerate, window=window,
                        transition_width=transition_width, ntaps=ntaps, gain=gain)
        return self.convolve(taps)
    def quantize(self, thresholds):
        levels = [float(v) for v in thresholds]
        levels.sort()
        nlevels = len(levels)
        output = np.zeros(self.nsamples, dtype=np.int16)
        compare = np.empty(self.nsamples, dtype=np.bool)
        mask = np.zeros(self.nsamples, dtype=np.bool)
        mask[:] = True
        
        for index in range(nlevels):
            np.less_equal(self.samples, levels[index], compare)
            np.logical_and(mask, compare, compare)
            output[compare] = index
            mask[compare] = False
        output[mask]  = nlevels
        return sampled_waveform(output, framerate=self.framerate)
    def play(self, gain=None):
        play(self.samples, self.framerate, gain=gain)
    def noise(self, distribution='normal', amplitude=1.0, loc=0.0, scale=1.0):
        return self + self.noise(self.nsamples, self.framerate, distribution=distribution,
                            amplitude=amplitude, loc=loc, scale=scale)
    def dft(self):
        assert self.domain == 'time', 'dft: can only apply to time-domain waveform'
        return sampled_waveform(np.fft.fft(self.samples), framerate=self.framerate,
                                domain='frequency')
    def idft(self):
        assert self.domain == 'frequency', 'idft: can only apply to freq-domain waveform'
        return sampled_waveform(np.fft.ifft(self.samples), framerate=self.framerate,
                                domain='time')
    def delay(self, nsamples=None, delta_t=None):
        assert self.domain == 'time', "delay: can only apply to time-domain waveform"
        assert (nsamples is not None or delta_t is None) or (nsamples is None or delta_t is not None),\
            'delay: exactly one of nsamples and delta must be specified'
        if nsamples is None:
            nsamples = int(float(self.framerate) / delta_t)
        result = np.copy(self.samples)
        if nsamples > 0:
            result[nsamples:] = self.samples[:-nsamples]
            result[0:nsamples] = self.samples[-nsamples:]
        return sampled_waveform(result, framerate=self.framerate)
    
class sinusoid(sampled_waveform):
    def __init__(self,nsamples=256e3,hz=1000,framerate=256e3,
                 amplitude=1.0,phase=0.0):
        assert hz <= framerate/2,"hz cannot exceed %gHz" % (framerate/2)
        phase_step = (2*np.pi*hz)/framerate
        temp = np.arange(nsamples,dtype=np.float64) * phase_step + phase
        np.cos(temp,temp)
        np.multiply(temp,amplitude,temp)
        sampled_waveform.__init__(self,temp,framerate=framerate)
        return list(sampled_waveform)
    def iter(self):
        return iter(self)
    ### TO DO:
        ##### 1) def plot(self,)
        ##### 2) def __iter__(self,)
        ##### 3) def ...
''' class array_waveform:
    def __init__(self, freq, length, framerate, phase):
        from scipy.signal import waveforms
        self.freq = 440.0
        self.length = 0.5
        self.framerate = 48000
        self.phase = 0.0
    def make_wave_data(freq, length, framerate, phase):
        length = int(length * framerate)
        phase *= float(framerate) / 2
        factor = float(freq) * (np.pi * 2) * factor
        return (np.arange(length) + phase) * factor
    def sine_wave(freq, length, framerate=48000, phase=0.0):
        data = make_wave_data(freq, length, framerate, phase)
        return np.sin(data)
    def sawtooth_wave(freq, length, framerate, phase):
        data = make_wave_data(freq, length, framerate, phase)
        return waveforms.sawtooth(data)
    def square_wave(freq, length, framerate, phase):
        data = generate_wave_input(freq, length, rate, phase)
        return waveforms.sawtooth(data)
class generator_waveform:
    def __init__(self, np.float64):
            freq = np.float64(freq, dtype=np.float64, copy=True)
        self.freq = np.float64(freq, copy=True)
        self.amplitude = amplitude
        #self.amplitude = 0.0
        self.framerate = framerate
        #self.framerate = 44100
        #self.framerate = 8000
        self.kind = kind
    def sine_wave(freq, framerate, amplitude):
        t = int(framerate / freq)
        if amplitude > 1.0: amplitude = 1.0
        if amplitude < 0.0: amplitude = 0.0
        lookup_table = [float(amplitude) * np.sin(2.0 * np.pi * float(freq) *
                              (float(i%t) / float(framerate))) for i in range(t)]
        result = (lookup_table[i%t] for i in it.count(0))
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
        result = (float(amplitude) * random.uniform(-1, 1) for i in it.count(0))
        return result
    def compute_samples(nchannels, nsamples):
        return slice(zip(*(map(sum, zip(*nchannels)) for channel in nchannels)), nsamples)
    def save_wav(filename, framerate=48000, buffer_size=2048):
        wav_file = wave.open(filename, 'w')
        nchannels = 1
        sampwidth = 2
        nsamples = len(samples)
        comptype = 'NONE'
        compname = 'not compressed'
        wav_file.setparams((nchannels, sampwidth, framerate, nframes, 
                            comptype, compname))
        for sample in np.array([iter(samples)] * buffer_size).T:
            samples = ''.join(''.join(struct.pack('h', int(sample * 32767)) for sample in nchannels) for channels in sample if nchannels is not None)
        wav_file.close()
        return
    def config():
        import sys
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
'''
def sin(nsamples=256e3,hz=1000,framerate=256e3,phase=0.0):
    return sinusoid(nsamples=nsamples,hz=hz,framerate=framerate,phase=-np.pi/2+phase)
