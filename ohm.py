# -*- coding: utf-8 -*-
"""
Created on Sat Sep 29 12:54:18 2018

@author: samla
"""

import pyaudio, wave, struct, time
#import board, busio
#import adafruit_bno055 as bno
import numpy as np
import pandas as pd

#i2c = busio.I2C(board.SCL, board.SDA)
#sensor = bno.BNO055(i2c)

accelerometer = []
magnetometer = []
gyroscope = []

frame_rate = 44100
sine_list = None

def get_sensor_AMG(sensor):
    global accelerometer, magnetometer, gyroscope
    while True:
        accelerometer.append(sensor.accelerometer)
        magnetometer.append(sensor.magnetometer)
        gyroscope.append(sensor.gyroscope)
        sensor_data = pd.DataFrame([accelerometer, magnetometer, gyroscope])
        return sensor_data
    else:
        accelerometer, magnetometer, gyroscope = None

        return accelerometer, magnetometer, gyroscope

def make_sine_list(freq, frame_rate=44100, dur=500, vol=.5):
    global sine_list
    nsamples = dur * (frame_rate / 1000.0)
    sine_list = [(vol * np.sin(2*np.pi*freq*(x/frame_rate))) for x in range(int(nsamples))]
    
    return sine_list

def save_list(file_name, file_contents, frame_rate=44100):
    wav_file = wave.open(file_name, 'w')
    
    nchannels = 1
    sampwidth = 2
    nframes = len(file_contents)
    comptype = 'NONE'
    compname = 'not compressed'
    
    wav_file.setparams((nchannels, sampwidth, frame_rate, nframes, comptype, compname))
    
    for sample in file_contents:
        wav_file.writeframes(struct.pack('h', int(sample * 32767.0)))
    
    wav_file.close()
    
    return file_name

def play_file(file_name):
    #define stream chunk   
    chunk = 1024  

    #open a wav format music  
    f = wave.open(file_name, 'rb')  
    #instantiate PyAudio  
    p = pyaudio.PyAudio()  
    #open stream  
    stream = p.open(format=p.get_format_from_width(f.getsampwidth()),  
                    channels=f.getnchannels(), rate=f.getframerate(),  
                    output=True)
    #read data  
    data = f.readframes(chunk)  

    #play stream  
    while len(data) > 0:  
        stream.write(data)  
        data = f.readframes(chunk)  

    #stop stream  
    stream.stop_stream()  
    stream.close()  

    #close PyAudio  
    p.terminate()
    
    return

#sine = make_sine_list(440.0)
#save_list('sine.wav', sine)
#play_stream('sine.wav')
#sine = make_sine_list(260.0)
#save_list('sine.wav', sine)
#play_stream('sine.wav')
    
pa = None
s = None

def init_audio(rate=44100):
    global pa, s
    pa = pyaudio.PyAudio()
    s = pa.open(format=pyaudio.paInt16, channels=1, rate=rate, output=True)
    return

def close_audio():
    global pa, s
    s.close()
    pa.terminate()
    return

def note(freq, length, amp=5000, rate=44100):
    t = np.linspace(0, length, length*rate)
    data = np.sin(2*np.pi*freq*t)*amp
    return data.astype(np.int16)

def tone(freq=440.0, length=0.5, amp=5000, rate=44100):
    global s
    tone = note(freq, length, amp, rate)
    s.write(tone)
    return

def main():
    init_audio()
    while True:
        #insert AMG here
        tone()
        time.sleep(.5)
    return

if __name__ == '__main__':
    main()
    
#for n in scale_list:
    #tone(n)