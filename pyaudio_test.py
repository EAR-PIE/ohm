import pyaudio, time
import numpy as np
import matplotlib.pyplot as plt
import scipy.signal as signal

channels = 1
rate = 44100

p = pyaudio.PyAudio()
full_data = np.array([])
dry_data = np.array([])

def main():
	stream = p.open(format=pyaudio.paFloat32,
			channels=channels,
			rate=rate,
			output=True,
			input=True,
			stream_callback=callback)
	stream.start_stream()
    while stream.is_active():
        time.sleep(10)
        stream.stop_stream()
        stream.close()

np_data = np.hstack(full_data)
plt.plot(np_data)
plt.title('Wet')
plt.show()

np_data = np.hstack(dry_data)
plt.plot(np_data)
plt.title('Dry')
plt.show()

p.terminate()

def callback(in_data, frame_count, time_info, flag):
	global b, a, full_data, dry_data, frames
	audio_data = np.fromstring(in_data, dtype=np.float32)
	dry_data = np.append(dry_data, audio_data)
	return(audio_data, pyaudio.paContinue)
def main():
	pass
