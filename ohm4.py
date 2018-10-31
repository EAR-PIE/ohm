# -*- coding: utf-8 -*-
"""
Created on Mon Oct 29 16:07:50 2018

@author: samla
"""

import numpy as np
import pandas as pd
import datetime as dt
import pyaudio, wave
#import board, busio
#import adafuit_bno055 as bno
from time import sleep

#i2c = busio.I2C(board.SCL, board.SDA); sensor = bno.BNO055(i2c);

class Sensor(object):
    #i2c = busio.I2C(board.SCL, board.SDA)
    #sensor = bno.BNO055(i2c)
    def __init__(self):
        self.sensor = sensor
        def get_sensor_dataframe():
            ''' returns a DataFrame of timestamped sensor data in tuples for 
                acceleration, gravity and gyroscope. '''
            global sensor
            while sensor:
                df = pd.DataFrame({'Speed': list(sensor.accelerometer),
                                   'Gravity': list(sensor.gravity),
                                   'Balance': list(sensor.gyroscope)})
                df.assign(timestamp=dt.datetime.now().time())
                sleep(0.25)
            return df
        
p = None; s = None;
def init_audio():
    global p, s
    p = pyaudio.PyAudio()
    s = p.open(format=pyaudio.paFloat32, channels=2, rate=48000, output=1)
    return
def close_audio():
    global p, s
    s.close()
    p.terminate()
    return
def play(waveform):
    global s
    s.write(waveform)
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
    return
    

class Waveform:
    def __init__(self, freq, length, framerate=48000, phase=0.0):
        self.freq = float(freq)
        self.length = float(length)
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
        return data.astype(np.float32)
    def _sawtooth(t):
        ''' wrapper for sawtooth wave '''
        tmod = np.mod(t, 2 * np.pi)
        return (tmod / np.pi) - 1
    def sawtooth(freq, length, framerate=48000, phase=0.0):
        ''' generates a sawtooth wave '''
        data = Waveform.input_for_wave(freq, length)
        return Waveform._sawtooth(data)
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

class Scale:
    def __init__(self, start_note, asc=True):
        self.start_note = start_note
        self.asc = asc
    def hz_progress(fixed_note, steps):
        #https://pages.mtu.edu/~suits/NoteFreqCalcs.html
        a = 2 ** (1/12)
        return fixed_note * a ** steps
    def major(start_note, asc=True):
        ''' returns a major scale given starting frequency, descending 
            or ascending is the second argument '''
        scale = [start_note, ]
        if asc == True:
            next_note = Scale.hz_progress(start_note, 2)
            scale.append(next_note)
            while next_note < start_note * 2:
                if len(scale) % 3 == 0 or len(scale) == 8:
                    next_note = Scale.hz_progress(next_note, 1)
                    scale.append(next_note)
                elif len(scale) > 7:
                    break
                else:
                    next_note = Scale.hz_progress(next_note, 2)
                    scale.append(next_note)
        elif asc == False:
            scale = [start_note, ]
            next_note = Scale.hz_progress(start_note, -1)
            scale.append(next_note)
            while next_note > start_note / 2:
                if len(scale) % 5 == 0:
                    next_note = Scale.hz_progress(next_note, -1)
                    scale.append(next_note)
                elif len(scale) > 7:
                    break
                else:
                    next_note = Scale.hz_progress(next_note, -2)
                    scale.append(next_note)
        return scale
    def natural_minor(start_note, asc=True):
        ''' same as major() but produces a natural minor scale '''
        scale = [start_note, ]
        if asc == True:
            next_note = Scale.hz_progress(start_note, 2)
            scale.append(next_note)
            while next_note < start_note * 2:
                if len(scale) % 2 == 0 and len(scale) % 4 != 0 and len(scale) != 6 or len(scale) == 5:
                    next_note = Scale.hz_progress(next_note, 1)
                    scale.append(next_note)
                elif len(scale) > 7:
                    break
                else:
                    next_note = Scale.hz_progress(next_note, 2)
                    scale.append(next_note)
        elif asc == False:
            next_note = Scale.hz_progress(start_note, -2)
            scale.append(next_note)
            while next_note > start_note / 2:
                if len(scale) == 3 or len(scale) == 6:
                    next_note = Scale.hz_progress(next_note, -1)
                    scale.append(next_note)
                elif len(scale) > 7:
                    break
                else:
                    next_note = Scale.hz_progress(next_note, -2)
                    scale.append(next_note)
        return scale
    def harmonic_minor(start_note, asc=True):
        ''' produces a harmonic minor scale given starting note and descending or
            ascending is specified '''
        scale = [start_note, ]
        if asc == True:
            next_note = Scale.hz_progress(start_note, 2)
            scale.append(next_note)
            while next_note < start_note * 2:
                if len(scale) % 2 == 0 and len(scale) % 4 != 0 and len(scale) != 6 or len(scale) == 5 or len(scale) == 7:
                    next_note = Scale.hz_progress(next_note, 1)
                    scale.append(next_note)
                elif len(scale) == 6:
                    next_note = Scale.hz_progress(next_note, 3)
                    scale.append(next_note)
                elif len(scale) > 7:
                    break
                else:
                    next_note = Scale.hz_progress(next_note, 2)
                    scale.append(next_note)
        elif asc == False:
            next_note = Scale.hz_progress(start_note, -1)
            scale.append(next_note)
            while next_note > start_note / 2:
                if len(scale) == 2:
                    next_note = Scale.hz_progress(next_note, -3)
                    scale.append(next_note)
                elif len(scale) == 3 or len(scale) == 6:
                    next_note = Scale.hz_progress(next_note, -1)
                    scale.append(next_note)
                elif len(scale) > 7:
                    break
                else:
                    next_note = Scale.hz_progress(next_note, -2)
                    scale.append(next_note)
        return scale
    def melodic_minor(start_note, asc=True):
        ''' produces a melodic minor scale, like the others... '''
        scale = [start_note, ]
        if asc == True:
            next_note = Scale.hz_progress(start_note, 2)
            scale.append(next_note)
            while next_note < start_note * 2:
                if len(scale) % 2 == 0 and len(scale) % 4 != 0 and len(scale) != 6 or len(scale) == 5 or len(scale) == 7:
                    next_note = Scale.hz_progress(next_note, 1)
                    scale.append(next_note)
                elif len(scale) == 6:
                    next_note = Scale.hz_progress(next_note, 3)
                    scale.append(next_note)
                elif len(scale) > 7:
                    break
                else:
                    next_note = Scale.hz_progress(next_note, 2)
                    scale.append(next_note)
        elif asc == False:
            next_note = Scale.hz_progress(start_note, -2)
            scale.append(next_note)
            while next_note > start_note / 2:
                if len(scale) == 3:
                    next_note = Scale.hz_progress(next_note, -3)
                    scale.append(next_note)
                elif len(scale) == 6:
                    next_note = Scale.hz_progress(next_note, -1)
                    scale.append(next_note)
                elif len(scale) > 7:
                    break
                else:
                    next_note = Scale.hz_progress(next_note, -2)
                    scale.append(next_note)
        return scale