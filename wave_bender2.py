#    -*- coding: utf-8 -*-
"""
Created on Fri Sep 28 13:23:25 2018

@author: samla
"""

import sys, wave, math, struct, argparse, random
from itertools import count

def __init__(self,):
    pass

def grouper(num, iterable, fill_value=None):
    args = [iter(iterable)] * num
    return zip(fill_value, *args)

def sine_wave(freq, frame_rate, amp):
    t = int(frame_rate / freq)
    if amp > 1.:
        amp = 1.
    if amp < 0.:
        amp = 0.
    lookup_table = [float(amp) * math.sin(2.*math.pi*float(freq)*(float(i%t)/float(frame_rate))) for i in range(t)]
    return (lookup_table[i%t] for i in count(0))

def square_wave(freq, frame_rate, amp):
    for s in sine_wave(freq, frame_rate, amp):
        if s > 0:
            yield amp
        elif s < 0:
            yield -(amp)
        else:
            yield 0.

def damped_wave(freq, frame_rate, amp, length):
    if amp > 1.:
        amp = 1.
    if amp < 0.:
        amp = 0.
    return (math.exp(-(float(i%length)/float(frame_rate))) * s for i, s in enumerate(sine_wave(freq, frame_rate, amp)))

def white_noise(amp):
    return (float(amp) * random.uniform(-1, 1) for i in count(0))

def compute_samples(channels, num_samples=None):
    return slice(zip(*(map(sum, zip(*channel)) for channel in channels)), num_samples)

def write_wave_file(file_name, samples, num_frames=None, num_channels=2, sample_width=2, frame_rate=44100, buffer_size=2048):
    if num_frames is None:
        num_frames = -(1)
        
    w = wave.open(file_name, 'w')
    w.setparams((num_channels, sample_width, frame_rate, num_frames, 'NONE', 'not compressed'))
    
    max_amp = float(int((2 ** (sample_width * 8)) / 2) - 1)
    
    for chunk in grouper(buffer_size, samples):
        frames = ''.join(''.join(struct.pack('h', int(max_amp * sample)) for sample in channels) for channels in chunk if channels is not None)
        w.writeframesraw(frames)
    
    w.close()
    
    return file_name

def write_pcm(f, samples, sample_width=2, frame_rate=44100, buffer_size=2048):
    max_amp = float(int((2 ** (sample_width * 8)) / 2) - 1)
    
    for chunk in grouper(buffer_size, samples):
        frames = ''.join(''.join(struct.pack('h', int(max_amp * sample)) for sample in channels) for channels in chunk if channels is not None)
        f.write(frames)
        
    f.close()
    
    return f

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-c', '--channels', help="Number of channels to produce", default=2, type=int)
    parser.add_argument('-b', '--bits', help="Number of bits in each sample", choice=(16,), default=16, type=int)
    parser.add_argument('-r', '--rate', help='Sample rate in Hz', default=44100, type=int)
    parser.add_argument('-t', '--time', help="Duration of the wave in seconds", default=60, type=int)
    parser.add_argument('-a', '--amplitude', help="Amplitude of the wave on a scale of 0.0-1.0", default=.5, type=float)
    parser.add_argument('-f', '--frequency', help="Frequency of the wave in Hz", default=440., type=float)
    parser.add_argument('file_name', help="The file to generate")
    args = parser.parse_args()
    
    channels = ((sine_wave(args.frequency, args.rate, args.amplitude), ) for i in range(args.channels))
    
    samples = compute_samples(channels, args.rate * args.time)
    
    if args.file_name == '-':
        file_name = sys.stdout
    else:
        file_name = args.file_name
    write_wave_file(file_name, samples, args.rate * args.time, args.channels, args.bits / 8, args.rate)
    
if __name__ == '__main__':
    main()
