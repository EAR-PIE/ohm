# -*- coding: utf-8 -*-
"""
date = 'October_2018'
author = 'sam_lax'
"""
#IMPORT PACKAGES:
    #for RPi's GPIO connected BNO055 sensor...
#import board, busio, adafruit_bno055 as bno
    #for interfacing with and outputting audio...
import pyaudio, time
    #for non-basic mathmatical operations and functions...
import numpy as np

#SET GLOBAL VARIABLES:
    #for PyAudio...
p = None; s = None;
    #for Board, BusIO and bno...
#i2c = busio.I2C(board.SCL, board.SDA); sensor = bno.BNO055(i2c);

#DEFINE FUNCTIONS:
    #create ability to initialize, generate and output audio...
def init_audio(rate=44100):
    global p, s
    p = pyaudio.PyAudio() #create PyAudio Object,
    s = p.open(format=pyaudio.paInt16, channels=1, rate=rate, output=True) #stream PyAudio Object with given parameters,
    return
def close_audio(): #uses attributes of PyAudio objects to turn them off,
    global p, s
    s.close()
    p.terminate()
    return
def note(freq, length, amp=5000, rate=44100): #IDEA! --> Fire / temp sensor adjusts the frame rate <--
    t = np.linspace(0, length, length*rate) #create an array to represent time of one frame,
    data = np.sin(2*np.pi*freq*t)*amp #compute sine wave,
    return data.astype(np.int16) #get the Hz value,
def play(freq=440, length=0.5, amp=5000, rate=44100):
    global s
    tone = note(freq, length, amp, rate) #set tone equal to Hz value,
    s.write(tone) #write tone to audio stream.
    return

def hz_stepper(fixed_note, steps):
    a = 2**(1/12)
    return fixed_note * a**steps

def major(start_note):
    scale = [start_note, ]
    next_note = hz_stepper(start_note, 2)
    scale.append(next_note)
    while next_note < start_note * 2:
        if len(scale) % 3 == 0 or len(scale) % 8 == 0:
            next_note = hz_stepper(next_note, 1)
            scale.append(next_note)
        elif len(scale) > 7:
            break
        else:
            next_note = hz_stepper(next_note, 2)
            scale.append(next_note)
    return scale
def natural_minor(start_note):
    scale = [start_note, ]
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
    return scale
def harmonic_minor(start_note):
    scale = [start_note, ]
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
    return scale
def melodic_minor(start_note):
    pass