# -*- coding: utf-8 -*-
"""
Created on Sun Oct 28 20:04:37 2018

@author: samla
"""
import numpy as np
import pandas as pd
import datetime as dt
import music21 as m21
import pyaudio, wave
#import board, busio
#import adafuit_bno055 as bno
#from time import sleep
from decimal import Decimal
from scipy import signal

#initialize orientation sensor
#i2c = busio.I2C(board.SCL, board.SDA); sensor = bno.BNO055(i2c);

#def get_sensor_dataframe():
#    ''' returns a DataFrame of timestamped sensor data for acceleration (x, y, z),
#        gravity (x, y, z) and gyroscopic orientation (x, y, z). '''
#    global sensor
#    while sensor:
#        df = pd.DataFrame({'Speed': list(sensor.accelerometer),
#                       'Gravity': list(sensor.gravity),
#                       'Balance': list(sensor.gyroscope)})
#        df.assign(timestamp=dt.datetime.now().time())
#        sleep(0.25)
#    return df
#def mag_to_value():
#    mag_tuple = sensor.magnetometer
#    mag = Decimal(0, mag_tuple, -2)
#    return mag
#def grav_to_hz():
#    grav_tuple = sensor.gravity
#    grav = Decimal(0, grav_tuple, mag_to_value())
#    return grav
#def accel_to_length():
#    accel_tuple = sensor.accelerometer
#    accel = Decimal(0, accel_tuple, grav_to_hz())
#    return accel

#initialize audio objects
p = None; s = None;
def init_audio(framerate=48000, mono=False, format_type='int16'):
    ''' initializes PyAudio objects '''
    global p, s
    p = pyaudio.PyAudio()
    if format_type == 'int8': format_type = pyaudio.paInt8
    if format_type == 'int16': format_type = pyaudio.paInt16
    if format_type == 'int24': format_type = pyaudio.paInt24
    if format_type == 'int32': format_type = pyaudio.paInt32
    if format_type == 'float32': format_type = pyaudio.paFloat32
    if mono == False:
        s = p.open(format=format_type, channels=2, rate=framerate, output=1)
    elif mono == True:
        s = p.open(format=format_type, channels=1, rate=framerate, output=1)
    return
def close_audio():
    ''' closes PyAudio objects '''
    global p, s
    s.close()
    p.terminate()
    return
def play(sound):
    ''' outputs audio of a wave; a wave can be Hz in integer or float form '''
    global s
    s.write(sound)
    return

def save_wav(filename, data=None):
    file = wave.open(filename, 'wb')
    nchannels = 2;      sampwidth = 2
    framerate = 48000;  nframes = len(data)
    comptype = 'NONE';  compname = 'not compressed'
    file.setparams((nchannels, sampwidth, 
                    framerate, nframes, 
                    comptype, compname))
    file.writeframesraw(data)
    file.close()
def read_wav(filename):
    file = wave.open(filename, 'rb')
    nframes = file.getnframes()
    data = file.readframes(nframes)
    return data
def play_wav(filename):
    data = read_wav(filename)
    play(data)
def save_then_play_wav(filename, data=None, rewrite=False, rewrite_length=0.0):
    if rewrite == True:
        while rewrite_length > 0.0:
            save_wav(filename, data)
            play_wav(filename)
            rewrite_length -= 0.1
        return
    elif rewrite == False:
        save_wav(filename, data)
        play_wav(filename)
        return

class M21:
    def __init__(self, M21_Note):
        self.M21_Note = M21_Note
        return
    def Hz_df(M21_Note, timestamp=False):
        freq = M21_Note.pitch.frequency
        length = M21_Note.duration.quarterLength
        if timestamp == True:
            return pd.Series({'freq': freq, 'length': length, 
                              'timestamp': dt.datetime.now().time()})
        elif timestamp == False:
            return pd.Series({'freq': freq, 'length': length})
        return
    def Hz_object(M21_Note):
        freq = M21_Note.pitch.frequency
        length = M21_Note.duration.quarterLength
        return dict(freq: length)
class Waveform:
    def __init__(self, freq, length, amp, framerate=48000, phase=0.0):
        self.freq = float(freq)
        self.length = float(length)
        self.amp = float(amp)
        self.framerate = int(framerate)
        self.phase = float(phase)
    def input_for_wave(freq, length, framerate=48000, phase=0.0):
        ''' creates building block for waves to be formed off of '''
        length = int(length * framerate)
        t = np.arange(length) / float(framerate)
        omega = float(freq) * 2 * np.pi
        phase *= 2 * np.pi
        return omega * t + phase
    def sine(freq, length, framerate=48000, amp=5000):
        ''' generates a sine wave '''
        t = np.linspace(0, length, length * framerate)
        data = np.sin(2 * np.pi * freq * t) * amp
        return data.astype(np.int16)
    def sine_generator(freq, framerate=48000):
        sine_of = (freq * 2 * np.pi) / framerate
        sample_n = 0
        while True:
            yield np.sin(sine_of * sample_n)
            sample_n += 1
            if sample_n == framerate:
                break
            else:
                continue
    def pulse(freq, framerate=48000, duty=0.5):
        sample_n = 0
        cycle_length = framerate / freq
        pulse_length = cycle_length * duty
        while True:
            if (sample_n % cycle_length) < pulse_length:
                yield 1.0
            else:
                yield -1.0
            sample_n += 1
            if sample_n == framerate:
                break
            else:
                continue
    def irregular_sawtooth(min_range, max_range):
        import random
        waveform = np.concatenate(
                [signal.sawtooth(2 * np.pi * np.linspace(
                        0, 1, random.randrange(
                                min_range, max_range))) for _ in range(10)])
        return waveform
    def sawtooth_generator(freq, framerate=48000, duty=0.5):
        sample_n = 0
        cycle_length = framerate / float(freq)
        midpoint = cycle_length * duty
        ascend_length = midpoint
        descend_length = cycle_length - ascend_length
        while True:
            cycle_position = sample_n % cycle_length
            if cycle_length < midpoint:
                yield (2 * cycle_position / ascend_length) - 1.0
            else:
                yield 1.0 - (2 * (cycle_position - midpoint) / descend_length)
            sample_n += 1
            if sample_n == framerate:
                break
            else:
                continue
    def _sawtooth(t):
        ''' wrapper for sawtooth wave '''
        tmod = np.mod(t, 2 * np.pi)
        return (tmod / np.pi) - 1
    def sawtooth(freq, length, framerate=48000, phase=0.0):
        ''' generates a sawtooth wave '''
        data = Waveform.input_for_wave(freq, length)
        return Waveform._sawtooth(data)
    def sawtooth1(freq, length, framerate=48000, amp=1.0):
        t = np.linspace(0, length, length * framerate)
        data = signal.sawtooth(2 * np.pi * freq * t) * amp
        return data
    def _square(t, duty=0.5):
        ''' wrapper for square wave '''
        y = np.zeros(t.shape)
        tmod = np.mod(t, 2 * np.pi)
        mask = tmod < duty * 2 * np.pi
        np.place(y, mask, 1)
        np.place(y, (1 - mask), -1)
        return y
    def square(freq, length, rate=48000, phase=0.0):
        ''' generates a square wave '''
        data = Waveform.input_for_wave(freq, length, rate, phase)
        return Waveform._square(data)
    def triangle(freq, length, framerate=48000, phase=0.0):
        return np.absolute(Waveform.sawtooth(freq, length, framerate, phase))
    def triangle_generator(amp, framerate=48000):
        section = framerate // 4
        for direction in (1, -1):
            for i in range(section):
                yield i * (amp / section) * direction
            for i in range(section):
                yield (amp - (i * (amp / section))) * direction
    def ring_mod(waveform_list, length):
        #####TO DO:
            ##### 1) needs work in general
        for part in waveform_list:
            waveforms = Waveform.triangle(part, length)
        return np.array(waveforms)
    def white_noise(framerate=48000):
        import random
        count = 0
        while True:
            if count == framerate:
                break
            else:
                yield (random.random() * 2) - 1.0
            count += 1
    def silence(length, framerate=48000):
        data = Waveform.input_for_wave(0.0, length)
        return data
    

#https://pages.mtu.edu/~suits/NoteFreqCalcs.html
def hz_stepper(fixed_note, steps):
    ''' baic formula for frequencies of notes of the equal tempered scale '''
    a = 2**(1/12)
    return fixed_note * a**steps

class Scale:
    def __init__(self, start_note, asc=True, kind='n'):
        self.start_note = start_note
        self.asc = asc
        self.kind = kind
    def major(start_note, asc=True):
        ''' returns a major scale given starting frequency, descending or ascending
            is the second argument '''
        scale = [start_note, ]
        if asc == True:
            next_note = hz_stepper(start_note, 2)
            scale.append(next_note)
            while next_note < start_note * 2:
                if len(scale) % 3 == 0 or len(scale) == 8:
                    next_note = hz_stepper(next_note, 1)
                    scale.append(next_note)
                elif len(scale) > 7:
                    break
                else:
                    next_note = hz_stepper(next_note, 2)
                    scale.append(next_note)
        elif asc == False:
            scale = [start_note, ]
            next_note = hz_stepper(start_note, -1)
            scale.append(next_note)
            while next_note > start_note / 2:
                if len(scale) % 5 == 0:
                    next_note = hz_stepper(next_note, -1)
                    scale.append(next_note)
                elif len(scale) > 7:
                    break
                else:
                    next_note = hz_stepper(next_note, -2)
                    scale.append(next_note)
        return scale
    def minor(start_note, asc=True, kind='natural'):
        if kind == 'natural' or kind == 'n':
            scale = [start_note, ]
            if asc == True:
                next_note = hz_stepper(start_note, 2)
                scale.append(next_note)
                while next_note < start_note * 2:
                    if len(scale) % 2 == 0 and len(scale) % 4 != 0 and len(scale) != 6 or len(scale) == 5:
                        next_note = hz_stepper(next_note, 1)
                        scale.append(next_note)
                    elif len(scale) > 7:
                        break
                    else:
                        next_note = hz_stepper(next_note, 2)
                        scale.append(next_note)
            elif asc == False:
                next_note = hz_stepper(start_note, -2)
                scale.append(next_note)
                while next_note > start_note / 2:
                    if len(scale) == 3 or len(scale) == 6:
                        next_note = hz_stepper(next_note, -1)
                        scale.append(next_note)
                    elif len(scale) > 7:
                        break
                    else:
                        next_note = hz_stepper(next_note, -2)
                        scale.append(next_note)
        elif kind == 'harmonic' or kind == 'h':
            scale = [start_note, ]
            if asc == True:
                next_note = hz_stepper(start_note, 2)
                scale.append(next_note)
                while next_note < start_note * 2:
                    if len(scale) % 2 == 0 and len(scale) % 4 != 0 and len(scale) != 6 or len(scale) == 5 or len(scale) == 7:
                        next_note = hz_stepper(next_note, 1)
                        scale.append(next_note)
                    elif len(scale) == 6:
                        next_note = hz_stepper(next_note, 3)
                        scale.append(next_note)
                    elif len(scale) > 7:
                        break
                    else:
                        next_note = hz_stepper(next_note, 2)
                        scale.append(next_note)
            elif asc == False:
                next_note = hz_stepper(start_note, -1)
                scale.append(next_note)
                while next_note > start_note / 2:
                    if len(scale) == 2:
                        next_note = hz_stepper(next_note, -3)
                        scale.append(next_note)
                    elif len(scale) == 3 or len(scale) == 6:
                        next_note = hz_stepper(next_note, -1)
                        scale.append(next_note)
                    elif len(scale) > 7:
                        break
                    else:
                        next_note = hz_stepper(next_note, -2)
                        scale.append(next_note)
        elif kind == 'melodic' or kind == 'm':
            scale = [start_note, ]
            if asc == True:
                next_note = hz_stepper(start_note, 2)
                scale.append(next_note)
                while next_note < start_note * 2:
                    if len(scale) % 2 == 0 and len(scale) % 4 != 0 and len(scale) != 6 or len(scale) == 5 or len(scale) == 7:
                        next_note = hz_stepper(next_note, 1)
                        scale.append(next_note)
                    elif len(scale) == 6:
                        next_note = hz_stepper(next_note, 3)
                        scale.append(next_note)
                    elif len(scale) > 7:
                        break
                    else:
                        next_note = hz_stepper(next_note, 2)
                        scale.append(next_note)
            elif asc == False:
                next_note = hz_stepper(start_note, -2)
                scale.append(next_note)
                while next_note > start_note / 2:
                    if len(scale) == 3:
                        next_note = hz_stepper(next_note, -3)
                        scale.append(next_note)
                    elif len(scale) == 6:
                        next_note = hz_stepper(next_note, -1)
                        scale.append(next_note)
                    elif len(scale) > 7:
                        break
                    else:
                        next_note = hz_stepper(next_note, -2)
                        scale.append(next_note)
        return scale

class Rhythm:
    def __init__(self, bpm, time_sig, length):
        self.bpm = int(bpm)
        self.time_sig = tuple(time_sig)
        self.length = float(length)
    def rest(bpm, length):
        pass
    def silence():
        pass
