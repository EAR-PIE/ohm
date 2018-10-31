# -*- coding: utf-8 -*-
"""
Created on Tue Oct  2 23:16:23 2018

@author: samla
"""

import time, pyaudio
import numpy as np

pa = None
s = None

def init_audio(rate=8000):
    global pa, s
    print("init_audio: Creating PyAudio Object...")
    pa = pyaudio.PyAudio()
    print("init_audio: Opening Stream...")
    s = pa.open(output=True, channels=1, rate=rate, format=pyaudio.paInt16)
    print("init_audio: Audio Stream Initialized!")
    return

def close_audio():
    global pa, s
    print("close_audio: Closing Stream...")
    s.close()
    print("close_audio: Terminating PyAudio Object...")
    pa.terminate()
    return

def note(freq, length, amp=5000, rate=8000):
    t = np.linspace(0, length, length*rate)
    data = np.sin(2*np.pi*freq*t)*amp
    return data.astype(np.int16)

def tone(freq=440.0, tone_length=.5, amp=5000, rate=8000):
    global s
    tone = note(freq, tone_length, amp, rate)
    s.write(tone)
    return

def main():
    init_audio()
    print("tone.py main(): Start Playing!")
    while True:
        print("tone.py main(): tone() 440")
        tone()
        time.sleep(.5)
        print("tone.py main(): tone() 261")
        tone(261, 1)
        time.sleep(.5)
        print("tone.py main(): tone() 880")
        tone(880, 1)
        time.sleep(.5)
        
    return

if __name__ == '__main__':
    main()