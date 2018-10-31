#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Oct 23 18:00:53 2018

@author: samla
"""

#!/usr/bin/env python
import numpy,wave,os,sys,time
import matplotlib.pyplot as p

################################################################################
## helper functions
################################################################################

# copy over some defns from numpy and matplotlib as a convenience...
ion = p.ion
pi = numpy.pi
def show():
    if not p.isinteractive():
        p.show()

# helper function looks at range of values and returns a new range,
# an engineering prefix and the scale factor
def eng_notation(range):
    x = max(abs(range[0]),abs(range[1]))   # find largest value
    if x <= 1e-12:
        scale = 1e15
        units = "f"
    elif x <= 1e-9:
        scale = 1e12
        units = "p"
    elif x <= 1e-6:
        scale = 1e9
        units = "n"
    elif x <= 1e-3:
        scale = 1e6
        units = "u"
    elif x <= 1:
        scale = 1e3
        units = "m"
    elif x >= 1e9:
        scale = 1e-9
        units = "G"
    elif x >= 1e6:
        scale = 1e-6
        units = "M"
    elif x >= 1e3:
        scale = 1e-3
        units = "k"
    else:
        scale = 1
        units = ""
    return ((range[0]*scale,range[1]*scale),units,scale)

# set up for audio playback
audio = None
try:
    import ossaudiodev
    audio = 'oss'
except ImportError:
    try:
        import pyaudio
        audio = 'pyaudio'
    except ImportError:
        pass

def play(samples,sample_rate,gain=None):
    nsamples = len(samples)
    
    # compute appropriate gain if none supplied
    if gain is None:
        gain = 1.0/max(numpy.absolute(samples))

    # convert -1V to 1V waveform to signed 16-bit ints
    asamples = numpy.zeros(nsamples,dtype=numpy.int16)
    numpy.multiply(samples,32767.0 * gain,asamples)
    offset = 0   # where we are in sending data
    nremaining = nsamples

    # send audio samples to output device
    if audio == 'oss':
        stream = ossaudiodev.open('w')
        fmt,chans,rate = stream.setparameters(ossaudiodev.AFMT_S16_LE,1,int(sample_rate))
        assert rate==sample_rate,\
               "%g is not a supported sample rate for audio playback" % sample_rate

        while nremaining > 0:
            can_send = min(stream.obuffree(),nremaining)
            if can_send > 0:
                end = offset + can_send
                stream.write(asamples[offset:end].tostring())
                offset = end
                nremaining -= can_send
                    
        stream.close()

    elif audio == 'pyaudio':
        dev = pyaudio.PyAudio()
        devinfo = dev.get_default_output_device_info()
        assert dev.is_format_supported(sample_rate,
                                       output_device = devinfo['index'],
                                       output_channels = 1,
                                       output_format = pyaudio.paInt16),\
               "%g is not a supported sample rate for audio playback" % sample_rate
        stream = dev.open(int(sample_rate),1,pyaudio.paInt16,output=True)

        while nremaining > 0:
            can_send = min(stream.get_write_available(),nremaining)
            if can_send > 0:
                end = offset + can_send
                stream.write(asamples[offset:end].tostring(),num_frames=can_send)
                offset = end
                nremaining -= can_send

        stream.close()

    else:
        assert False,"Sorry, no audio device library could be found"

# returns list of sampled_waveforms, one per channel.
# Audio samples are in range -1.0 to +1.0 if gain=None is used
def read_wavfile(filename,gain=None):
    assert os.path.exists(filename),"file %s doesn't exist" % filename
    wav = wave.open(filename,'rb')
    nframes = wav.getnframes()
    assert nframes > 0,"%s doesn't have any audio data!" % filename
    nchan = wav.getnchannels()
    sample_rate = wav.getframerate()
    sample_width = wav.getsampwidth()

    # see http://ccrma.stanford.edu/courses/422/projects/WaveFormat/
    g = 1.0 if gain is None else gain
    if sample_width == 1:
        # data is unsigned bytes, 0 to 255
        dtype = numpy.uint8
        scale = g / 127.0
        offset = -1.0
    elif sample_width == 2:
        # data is signed 2's complement 16-bit samples (little-endian byte order)
        dtype = numpy.int16
        scale = g / 32767.0
        offset = 0.0
    elif sample_width == 4:
        # data is signed 2's complement 32-bit samples (little-endian byte order)
        dtype = numpy.int32
        scale = g / 2147483647.0
        offset = 0.0
    else:
        assert False,"unrecognized sample width %d" % sample_width

    outputs = [numpy.zeros(nframes,dtype=numpy.float64)
               for i in xrange(nchan)]

    count = 0
    while count < nframes:
        audio = numpy.frombuffer(wav.readframes(nframes-count),dtype=dtype)
        end = count + (len(audio) / nchan)
        for i in xrange(nchan):
            outputs[i][count:end] = audio[i::nchan]
        count = end
        
    # scale data appropriately
    for i in xrange(nchan):
        numpy.multiply(outputs[i],scale,outputs[i])
        if offset != 0: numpy.add(outputs[i],offset,outputs[i])

    # apply auto gain
    if gain is None:
        maxmag = max([max(numpy.absolute(outputs[i])) for i in xrange(nchan)])
        for i in xrange(nchan):
            numpy.multiply(outputs[i],1.0/maxmag,outputs[i])

    return [sampled_waveform(outputs[i],sample_rate=sample_rate)
            for i in xrange(nchan)]

# write a n-channel .wav file using samples fron the n supplied waveforms
def write_wavfile(*waveforms,**keywords):
    filename = keywords.get('filename',None)
    gain = keywords.get('gain',None)
    sample_width = keywords.get('sample_width',2)

    assert filename,"filename must be specified"
    nchan = len(waveforms)
    assert nchan > 0,"must supply at least one waveform"
    nsamples = waveforms[0].nsamples
    sample_rate = waveforms[0].sample_rate
    domain = waveforms[0].domain
    for i in xrange(1,nchan):
        assert waveforms[i].nsamples==nsamples,\
               "all waveforms must have the same number of samples"
        assert waveforms[i].sample_rate==sample_rate,\
               "all waveforms must have the same sample rate"
        assert waveforms[i].domain==domain,\
               "all waveforms must have the same domain"

    if gain is None:
        maxmag = max([max(numpy.absolute(waveforms[i].samples))
                      for i in xrange(nchan)])
        gain = 1.0/maxmag

    if sample_width == 1:
        dtype = numpy.uint8
        scale = 127.0 * gain
        offset = 127.0
    elif sample_width == 2:
        dtype = numpy.int16
        scale = 32767.0 * gain
        offset = 0
    elif sample_width == 4:
        dtype = numpy.int32
        scale = 2147483647.0 * gain
        offset = 0
    else:
        assert False,"sample_width must be 1, 2, or 4 bytes"

    # array to hold scaled data for 1 channel
    temp = numpy.empty(nsamples,dtype=numpy.float64)
    # array to hold frame data all channels
    data = numpy.empty(nchan*nsamples,dtype=dtype)

    # process the data
    for i in xrange(nchan):
        # apply appropriate scale and offset
        numpy.multiply(waveforms[i].samples,scale,temp)
        if offset != 0: numpy.add(temp,offset,temp)
        # interleave channel samples in the output array
        data[i::nchan] = temp[:]

    # send frames to wav file
    wav = wave.open(filename,'wb')
    wav.setnchannels(nchan)
    wav.setsampwidth(sample_width)
    wav.setframerate(sample_rate)
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
        return [0.53836 - 0.46164*numpy.cos((2*numpy.pi*i)/order)
                for i in xrange(ntaps)]
    elif window == 'hann' or window == 'hanning':
        return [0.5 - 0.5*numpy.cos((2*numpy.pi*i)/order)
                for i in xrange(ntaps)]
    elif window == 'bartlett':
        return [1.0 - abs(2*i/order - 1)
                for i in xrange(ntaps)]
    elif window == 'blackman':
        alpha = .16
        return [(1-alpha)/2 - 0.50*numpy.cos((2*numpy.pi*i)/order)
                - (alpha/2)*numpy.cos((4*numpy.pi*i)/order)
                for i in xrange(ntaps)]
    elif window == 'nuttall':
        return [0.355768 - 0.487396*numpy.cos(2*numpy.pi*i/order)
                         + 0.144232*numpy.cos(4*numpy.pi*i/order)
                         - 0.012604*numpy.cos(6*numpy.py*i/order)
                for i in xrange(ntaps)]
    elif window == 'blackman-harris':
        return [0.35875 - 0.48829*numpy.cos(2*numpy.pi*i/order)
                        + 0.14128*numpy.cos(4*numpy.pi*i/order)
                        - 0.01168*numpy.cos(6*numpy.pi*i/order)
                for i in xrange(ntaps)]
    elif window == 'blackman-nuttall':
        return [0.3635819 - 0.4891775*numpy.cos(2*numpy.pi*i/order)
                          + 0.1365995*numpy.cos(4*numpy.pi*i/order)
                          - 0.0106411*numpy.cos(6*numpy.py*i/order)
                for i in xrange(ntaps)]
    elif window == 'flat top':
        return [1 - 1.93*numpy.cos(2*numpy.pi*i/order)
                  + 1.29*numpy.cos(4*numpy.pi*i/order)
                  - 0.388*numpy.cos(6*numpy.py*i/order)
                  + 0.032*numpy.cos(8*numpy.py*i/order)
                for i in xrange(ntaps)]
    elif window == 'rectangular' or window == 'dirichlet':
        return [1 for i in xrange(ntaps)]
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
        wc = [2*numpy.pi*fc[i] for i in (0,1)]
    else:
        fc = float(cutoff)/sample_rate
        wc = 2*numpy.pi*fc

    if type == 'low-pass':
        # for low pass, gain @ DC = 1.0
        for i in xrange(ntaps):
            if i == middle:
                coeff = (wc/numpy.pi) * window[i]
                fmax += coeff
            else:
                n = i - middle
                coeff = (numpy.sin(n*wc)/(n*numpy.pi)) * window[i]
                fmax += coeff
            taps[i] = coeff
    elif type == 'high-pass':
        # for high pass gain @ nyquist freq = 1.0
        for i in xrange(ntaps):
            if i == middle:
                coeff = (1.0 - wc/numpy.pi) * window[i]
                fmax += coeff
            else:
                n = i - middle
                coeff = (-numpy.sin(n*wc)/(n*numpy.pi)) * window[i]
                fmax += coeff * numpy.cos(n*numpy.pi)
            taps[i] = coeff
    elif type == 'band-pass':
        # for band pass gain @ (fc_lo + fc_hi)/2 = 1.0
        # a band pass filter is simply the combination of
        #   a high-pass filter at fc_lo  in series with
        #   a low-pass filter at fc_hi
        # so convolve taps to get the effect of composition in series
        for i in xrange(ntaps):
            if i == middle:
                coeff = ((wc[1] - wc[0])/numpy.pi) * window[i]
                fmax += coeff
            else:
                n = i - middle
                coeff = ((numpy.sin(n*wc[1]) - numpy.sin(n*wc[0]))/(n*numpy.pi)) * window[i]
                fmax += coeff * numpy.cos(n*(wc[0] + wc[1])*0.5)
            taps[i] = coeff
    elif type == 'band-reject':
        # for band reject gain @ DC = 1.0
        # a band reject filter is simply the combination of
        #   a low-pass filter at fc_lo   in series with a
        #   a high-pass filter at fc_hi
        # so convolve taps to get the effect of composition in series
        for i in xrange(ntaps):
            if i == middle:
                coeff = (1.0 - ((wc[1] - wc[0])/numpy.pi)) * window[i]
                fmax += coeff
            else:
                n = i - middle
                coeff = ((numpy.sin(n*wc[0]) - numpy.sin(n*wc[1]))/(n*numpy.pi)) * window[i]
                fmax += coeff
            taps[i] = coeff
    else:
        assert False,"compute_taps: unrecognized filter type %s" % type

    gain = gain / fmax
    for i in xrange(ntaps): taps[i] *= gain
    return taps

def plot_freq(taps,title="Frequency response"):
    p.figure()
    p.title(title)
    omega_list = numpy.linspace(0,numpy.pi,num=501)
    H = [sum([taps[m]*numpy.exp(-1j*omega*m)
              for m in xrange(len(taps))])
         for omega in omega_list]
    p.plot(omega_list,numpy.absolute(H))
    p.xlabel("$\Omega$")
    p.grid()

################################################################################
## sampled_waveform base class
################################################################################

class sampled_waveform:
    def __init__(self,samples,sample_rate=1e6,domain='time'):
        if not isinstance(samples,numpy.ndarray):
            samples = numpy.array(samples,dtype=numpy.float,copy=True)
        self.samples = numpy.array(samples, copy=True)   # a numpy array
        self.nsamples = len(samples)
        self.sample_rate = sample_rate
        self.domain = domain

    def _check(self,other):
        if isinstance(other,(int,float,numpy.ndarray)):
            return other
        elif isinstance(other,(tuple,list)):
            return numpy.array(other)
        elif isinstance(other,sampled_waveform):
            assert self.nsamples == other.nsamples,\
                   "both waveforms must have same number of samples"
            assert self.sample_rate == other.sample_rate,\
                   "both waveforms must have same sample rate"
            assert self.domain == other.domain,\
                   "both waveforms must be in same domain"
            return other.samples
        else:
            assert False,"unrecognized operand type"

    def real(self):
        return sampled_waveform(numpy.real(self.samples),
                                sample_rate=self.sample_rate,
                                domain=self.domain)

    def imag(self):
        return sampled_waveform(numpy.imag(self.samples),
                                sample_rate=self.sample_rate,
                                domain=self.domain)

    def magnitude(self):
        return sampled_waveform(numpy.absolute(self.samples),
                                sample_rate=self.sample_rate)

    def angle(self):
        return sampled_waveform(numpy.angle(self.samples),
                                sample_rate=self.sample_rate)

    def __add__(self,other):
        ovalues = self._check(other)
        return sampled_waveform(self.samples + ovalues,
                                sample_rate=self.sample_rate,
                                domain=self.domain)

    def __radd__(self,other):
        ovalues = self._check(other)
        return sampled_waveform(self.samples + ovalues,
                                sample_rate=self.sample_rate,
                                domain=self.domain)

    def __sub__(self,other):
        ovalues = self._check(other)
        return sampled_waveform(self.samples - ovalues,
                                sample_rate=self.sample_rate,
                                domain=self.domain)

    def __rsub__(self,other):
        ovalues = self._check(other)
        return sampled_waveform(ovalues - self.samples,
                                sample_rate=self.sample_rate,
                                domain=self.domain)

    def __mul__(self,other):
        ovalues = self._check(other)
        return sampled_waveform(self.samples * ovalues,
                                sample_rate=self.sample_rate,
                                domain=self.domain)

    def __rmul__(self,other):
        ovalues = self._check(other)
        return sampled_waveform(self.samples * ovalues,
                                sample_rate=self.sample_rate,
                                domain=self.domain)

    def __div__(self,other):
        ovalues = self._check(other)
        return sampled_waveform(self.samples / ovalues,
                                sample_rate=self.sample_rate,
                                domain=self.domain)

    def __rdiv__(self,other):
        ovalues = self._check(other)
        return sampled_waveform(ovalues / self.samples,
                                sample_rate=self.sample_rate,
                                domain=self.domain)

    def __abs__(self):
        return sampled_waveform(numpy.absolute(self.samples),
                                sample_rate=self.sample_rate,
                                domain=self.domain)
        
    def __len__(self):
        return len(self.samples)

    def __mod__(self,other):
        ovalues = self._check(other)
        return sampled_waveform(numpy.fmod(self.samples,ovalues),
                                sample_rate=self.sample_rate,
                                domain=self.domain)


    def slice(self,start,stop,step=1):
        return sampled_waveform(self.samples[start:stop:step],
                                sample_rate=self.sample_rate,
                                domain=self.domain)

    def __getitem__(self,key):
        return self.samples.__getitem__(key)

    def __setitem__(self,key,value):
        if isinstance(value,sampled_waveform):
            value = value.samples
        self.samples.__setitem__(key,value)

    def __iter__(self):
        return self.samples.__iter__()

    def __str__(self):
        return str(self.samples)+('@%d samples/sec' % self.sample_rate)

    def resize(self,length):
        self.samples.resize(length)
        self.nsamples = length

    def convolve(self,taps):
        conv_res = numpy.convolve(self.samples,taps)
        offset = len(taps)/2
        return sampled_waveform(conv_res[offset:offset+self.nsamples],
                                sample_rate=self.sample_rate,
                                domain=self.domain)

    def modulate(self,hz,phase=0.0,gain=1.0):

        periods = float(self.nsamples*hz)/float(self.sample_rate)
        if abs(periods - round(periods)) > 1.0e-6:
            print("Warning: Non-integral number of modulation periods")
            print("nsamples=%d hz=%f sample_rate=%d periods=%f" % (self.nsamples, hz,self.sample_rate,periods))

        result = sinusoid(hz=hz,nsamples=self.nsamples,
                          sample_rate=self.sample_rate,phase=phase,
                          amplitude=gain)
        numpy.multiply(self.samples,result.samples,result.samples)
        return result

    def filter(self,type,cutoff,
               window='hamming',transition_width=None,ntaps=None,error=0.05,gain=1.0):
        if ntaps is None and transition_width is None:
            # ensure sufficient taps to represent a frequency resolution of error*cutoff
            ntaps = int(float(self.sample_rate)/(float(cutoff)*error))
            if ntaps & 1: ntaps += 1
        taps = fir_taps(type,cutoff,self.sample_rate,
                        window=window,transition_width=transition_width,
                        ntaps=ntaps,gain=gain)
        return self.convolve(taps)

    def quantize(self,thresholds):
        levels = [float(v) for v in thresholds]
        levels.sort()  # min to max
        nlevels = len(levels)
        output = numpy.zeros(self.nsamples,dtype=numpy.int)
        compare = numpy.empty(self.nsamples,dtype=numpy.bool)
        mask = numpy.zeros(self.nsamples,dtype=numpy.bool)
        mask[:] = True
        # work our way from min slicing level to max
        for index in xrange(nlevels):
            # find elements <= current slicing level
            numpy.less_equal(self.samples,levels[index],compare)
            # mask out those we've already categorized
            numpy.logical_and(mask,compare,compare)
            # set symbol value for outputs in this bucket
            output[compare] = index
            # now mark the newly bucketed values as processed
            mask[compare] = False
        # remaining elements are in last bucket
        output[mask] = nlevels
        return sampled_waveform(output,sample_rate=self.sample_rate)

    def play(self,gain=None):
        play(self.samples,self.sample_rate,gain=gain)

    def plot(self,xaxis=None,yaxis=None,title="",linetype="b",absplot=False):
        p.figure()
        p.title(title)
        if self.domain == 'time':
            x_range,x_prefix,x_scale = eng_notation((0,float(self.nsamples - 1)/self.sample_rate))
            if xaxis is None: xaxis = x_range
            else: xaxis = (xaxis[0]*x_scale,xaxis[1]*x_scale)
            x_step = (x_range[1] - x_range[0])/float(self.nsamples-1)
            p.plot(numpy.arange(self.nsamples)*x_step + x_range[0],self.samples,linetype)
            if yaxis is None:
                yaxis = (min(self.samples),max(self.samples))
                dy = yaxis[1]-yaxis[0]
                yaxis = (yaxis[0] - .1*dy,yaxis[1] + .1*dy)
            p.axis((xaxis[0],xaxis[1],yaxis[0],yaxis[1]))
            p.xlabel(x_prefix+'s')
            p.ylabel('V')
        elif self.domain == 'frequency':
            nyquist = self.sample_rate/2
            x_range,x_prefix,x_scale = eng_notation((-nyquist,nyquist))
            if xaxis is None:
                xaxis = x_range
            else:
                xaxis = (xaxis[0]*x_scale,xaxis[1]*x_scale)
            if (self.nsamples & 1) == 0:
                # even number of samples
                x_step = (x_range[1] - x_range[0])/float(self.nsamples)
            else:
                # odd number of samples
                x_step = (x_range[1] - x_range[0])/float(self.nsamples-1)

            dftscale = float(1.0)/float(self.nsamples)
            dft = dftscale * numpy.fft.fftshift(self.samples)
            if absplot:
                numpy.absolute(dft,dft)
                p.plot(numpy.arange(self.nsamples)*x_step + x_range[0],dft,linetype)
                if yaxis is None:
                    yaxis = (min(dft),max(dft))
                p.axis((xaxis[0],xaxis[1],yaxis[0],yaxis[1]))
                p.xlabel(x_prefix+'Hz')
                p.ylabel('Magnitude')
            else:
                p.subplot(211)
                p.title(title)
                K = (len(dft)-1)/2
                p.plot(numpy.arange(self.nsamples)*x_step+x_range[0],dft.real,"b.")
                p.plot(numpy.arange(self.nsamples)*x_step + x_range[0],dft.real)
                yaxis = (min(-0.1, min(dft.real)-0.05),max(0.1,max(dft.real)+0.05))
                p.axis((xaxis[0],xaxis[1],yaxis[0],yaxis[1]))
                p.xlabel(x_prefix+'Hz')
                #p.xlabel('$\Omega$')
                p.ylabel('Real')
                p.subplot(212)
                p.plot(numpy.arange(self.nsamples)*x_step + x_range[0],dft.imag,"b.")
                p.plot(numpy.arange(self.nsamples)*x_step + x_range[0],dft.imag)
                yaxis = (min(-0.1, min(dft.imag)-0.05),max(0.1,max(dft.imag)+0.05))
                p.axis((xaxis[0],xaxis[1],yaxis[0],yaxis[1]))
                p.xlabel(x_prefix+'Hz')
                #p.xlabel('$\Omega$')
                p.ylabel('Imag')

    def spectrum(self,xaxis=None,yaxis=None,title="",npoints=None):
        if npoints is None: npoints = 256000
        npoints = min(self.nsamples,npoints)
        waveform = self.slice(0,npoints)
        if waveform.domain == 'frequency':
            waveform.plot(xaxis=xaxis,yaxis=yaxis,title=title)
        else:
            waveform.dft().plot(xaxis=xaxis,yaxis=yaxis,title=title)

    def eye_diagram(self,samples_per_symbol,title="Eye diagram"):
        assert self.domain == 'time',"eye_diagram: only valid for time domain waveforms"
        p.figure()
        p.title(title)
        nright = self.nsamples - self.nsamples % (2*samples_per_symbol)
        # reshape samples into ? rows by 2*samples_per_symbol columns
        psamples = numpy.reshape(self.samples[:nright],(-1,2*samples_per_symbol))
        # plot plots 2D arrays column-by-column
        p.plot(psamples.T)

    def noise(self,distribution='normal',amplitude=1.0,loc=0.0,scale=1.0):
        return self + noise(self.nsamples,self.sample_rate,
                            distribution=distribution,
                            amplitude=amplitude,loc=loc,scale=scale)

    def dft(self):
        assert self.domain=='time',\
               'dft: can only apply to time domain waveforms'
        return sampled_waveform(numpy.fft.fft(self.samples),
                                sample_rate=self.sample_rate,
                                domain='frequency')

    def idft(self):
        assert self.domain=='frequency',\
           'idft: can only apply to frequency domain waveforms'
        return sampled_waveform(numpy.fft.ifft(self.samples),
                                sample_rate=self.sample_rate,
                                domain='time')

    def delay(self,nsamples=None,deltat=None):
        assert self.domain == 'time',\
               "delay: can only delay a time-domain waveform"
        assert (nsamples is not None or deltat is None) or \
               (nsamples is None or deltat is not None),\
               "delay: Exactly one of nsamples and delta must be specified"
        if nsamples is None:
            nsamples = int(float(self.sample_rate)/deltat)

        # Keep delayed result periodic
        result = numpy.copy(self.samples)
        if nsamples > 0:
            result[nsamples:] = self.samples[:-nsamples]
            result[0:nsamples] = self.samples[-nsamples:]
        return sampled_waveform(result,sample_rate=self.sample_rate)

    def resample(self,sample_rate,bw=None,ntaps=101,gain=1.0):
        nyquist = min(self.sample_rate,sample_rate)/2.0
        if bw is None: bw = nyquist
        else: bw = min(bw,nyquist)

        if self.sample_rate == sample_rate and bw == sample_rate/2.0:
            return self

        if self.sample_rate >= sample_rate:
            # decimation required
            fdecimation = float(wfs)/sample_rate
            decimation = int(fdecimation)
            assert sample_rate*decimation == self.sample_rate,\
                   "resample: required decimation (%g) must be an integer" % fdecimation
            # apply anti-aliasing filter
            samples = numpy.convolve(fir_taps('low-pass',bw,sample_rate,ntaps=ntaps,gain=gain),
                                     self.samples)
            if decimation != 1:
                samples = samples[::decimation]
        else:
            # interpolation required
            finterpolation = float(sample_rate)/wfs
            interpolation = int(finterpolation)
            assert self.sample_rate*interpolation == sample_rate,\
                   "resample: required interpolation (%g) must be an integer" % finterpolation
            samples = numpy.zeros(interpolation*self.nsamples,dtype=numpy.float64)
            samples[::interpolation] = self.samples
            # apply reconstruction filter
            samples = numpy.convolve(fir_taps('low-pass',bw,sample_rate,
                                              ntaps=ntaps,gain=gain*interpolation),
                                     samples)

        return sampled_waveform(samples,sample_rate=sample_rate,domain='time')

################################################################################
## sources
################################################################################

class sinusoid(sampled_waveform):
    def __init__(self,nsamples=256e3,hz=1000,sample_rate=256e3,
                 amplitude=1.0,phase=0.0):
        assert hz <= sample_rate/2,"hz cannot exceed %gHz" % (sample_rate/2)
        phase_step = (2*numpy.pi*hz)/sample_rate
        temp = numpy.arange(nsamples,dtype=numpy.float64) * phase_step + phase
        numpy.cos(temp,temp)
        numpy.multiply(temp,amplitude,temp)
        sampled_waveform.__init__(self,temp,sample_rate=sample_rate)

def sin(nsamples=256e3,hz=1000,sample_rate=256e3,phase=0.0):
    return sinusoid(nsamples=nsamples,hz=hz,sample_rate=sample_rate,phase=-numpy.pi/2+phase)

def cos(nsamples=256e3,hz=1000,sample_rate=256e3,phase=0.0):
    return sinusoid(nsamples=nsamples,hz=hz,sample_rate=sample_rate,phase=phase)

class csinusoid(sampled_waveform):
    def __init__(self,nsamples=256e3,hz=1000,sample_rate=256e3,
                 amplitude=1.0,phase=0.0):
        assert hz <= sample_rate/2,"hz cannot exceed %gHz" % (sample_rate/2)
        omega = (float(hz)/sample_rate) * 2.0 * numpy.pi
        temp = numpy.arange(0,nsamples) * omega + phase
        sampled_waveform.__init__(self,amplitude*numpy.exp(1j*temp),
                                  sample_rate=sample_rate)

# distribution 'normal', 'laplace', 'raleigh', 'uniform', 'triangular', 'impulse'
def noise(nsamples,sample_rate,distribution='normal',amplitude=1.0,loc=0.0,scale=1.0):
    if distribution in ('normal','gaussian'):
        noise = numpy.random.normal(loc,scale,size=nsamples)
    elif distribution == 'uniform':
        noise = numpy.random.uniform(loc-scale,loc+scale,size=nsamples)
    elif distribution == 'triangular':
        noise = numpy.random.triangular(loc-scale,loc,loc+scale,size=nsamples)
    elif distribution == 'impulse':
        # from gr_random.cc in the gnuradio code
        # scale: 5 => scratchy, 8 => geiger
        noise = numpy.random.uniform(size=nsamples)
        numpy.log(noise,noise)
        numpy.multiply(noise,-1.4142135623730951,noise)   # * -sqrt(2)
        noise[noise < scale] = 0.0
    elif distribution == 'laplace':
        noise = numpy.random.laplace(loc,scale,size=nsamples)
    elif distribution == 'raleigh':
        noise = numpy.random.raleigh(scale,size=nsamples)
    else:
        assert False,"unrecognized distribution %s" % distribution
    if amplitude != 1.0:
        numpy.multiply(noise,amplitude,noise)
    return sampled_waveform(noise,
                            sample_rate=sample_rate,
                            domain='time')

def wavfile(filename,gain=None,sample_rate=None,bw=None,nsamples=None):
    result = read_wavfile(filename,gain=gain)[0]
    if sample_rate:
        result = result.resample(sample_rate=sample_rate,bw=bw,ntaps=101)
    if nsamples:
        result.resize(nsamples)
    return result

################################################################################
## usrp interface
################################################################################

import threading

# read from usrp rx channel using a separate thread
class rx_helper(threading.Thread):
    def __init__(self,rx,ibuffer,qbuffer):
        threading.Thread.__init__(self)
        self.rx = rx
        self.ibuffer = ibuffer
        self.qbuffer = qbuffer
        self.nsamples = min(ibuffer.size,qbuffer.size)

    def run(self):
        # start receiver
        assert self.rx.start(),"usrp receiver failed to start"

        # receive data loop
        count = 0
        while count < self.nsamples:
            # compute how many samples to read on this call
            want = min((self.nsamples-count)<<2,131072)
            # read 'em
            buffer,overrun = self.rx.read(want)
            # deinterleave samples into I and Q buffers
            data = numpy.frombuffer(buffer,dtype=numpy.int16)
            got = data.size/2
            if got > 0:
                end = count + got
                self.ibuffer[count:end] = data[0::2]
                self.qbuffer[count:end] = data[1::2]
                count = end

        # done, stop receiver
        self.rx.stop()

# transmit to tx channel using a separate thread
class tx_helper(threading.Thread):
    def __init__(self,tx,samples):
        threading.Thread.__init__(self)
        self.tx = tx
        self.samples = samples
        self.nxmit = samples.size
        #print self.nxmit

    def run(self):
        # start transmitter
        assert self.tx.start(),"usrp transmitter failed to start"

        offset = 0   # where we are in transmitting data
        while offset < self.nxmit:
            end = min(offset+131072,self.nxmit)
            data = self.samples[offset:end].tostring()
            bytes_sent,underrun = self.tx.write(data)
            if underrun: print('usrp transmit: underrun...')
            offset += bytes_sent/2

        # wait until done, stop transmitter
        self.tx.wait_for_completion()
        self.tx.stop()

try:
    import usrp_602 as usrp

    # send a waveform through the IR channel and return what gets received
    # rcv = usrp_channel(xmit,...)
    def usrp_ir_channel(xmit,           # sampled_waveform to be transmitted
                        tx_gain=1.0,    # transmit scaling factor
                        tx_pga_gain_db=0, # usrp transmit gain
                        rx_gain=1.0,    # receive scaling factor
                        rx_pga_gain_db=0, # usrp receive gain
                        rx_adc_offset=0,  # receive adc offset
                        frequency=0,    # usrp up-convert/down-convert frequency
                        chan=1,         # assume IR board is plugged into "B" side of USRP
                        preamble=128*20,  # how many zero samples to transmit before data
                        postamble=128*20, # how many zero samples to transmit after data
                        extra=128*50       # number of additional receive samples
                        ):
        nsamples = xmit.nsamples
        sample_rate = xmit.sample_rate
        
        # make sure we have a USRP plugged in
        if not usrp.usrp_find_device(0):
            assert False,"usrp_ir_channel: can't find USRP board %s" % board


        ########
        # tx setup
        #######

        # interp rate updated below once we know dac rate
        tx = usrp.usrp_tx(500)
        tx.stop()

        # configure which DACs to use
        tx.set_mux(0x0098 if chan == 0 else 0x9800)

        # set correct interpolation rate
        interp = int(tx.dac_rate/sample_rate)

        assert interp <= 512,\
               "usrp_ir_channel: sorry, sample rate must be at least %g" % tx.dac_rate/512
        tx.set_interp_rate(interp)
        interp_rate = tx.interp_rate

        # set PGA gain
        assert tx_pga_gain_db >= tx.pga_min and tx_pga_gain_db <= tx.pga_max,\
               "usrp_ir_channel: tx_pga_gain_db must be in the range %g to %g" % (tx.pga_min,tx.pga_max)

        # DACs 0 and 1 share a gain setting, as do DACs 2 and 3
        tx.set_pga(2*chan,tx_pga_gain_db)

        # set frequency of DUC
        assert tx.set_tx_freq(chan,frequency),\
               "usrp_ir_channel: set_tx_freq failed with value %g" % frequency

	# We are only using one active channel!!! (jkw)
        assert tx.set_nchannels(1), \
               "usrp_ir_channel: set_tx_nchannel failed with value 1" 

        # adjust number of transmitted samples to be a multiple of 128 
        # since we have to transmit a multiple of 512 bytes and there are 
        # 4 bytes xmitted per sample
        nxmit = nsamples + preamble + postamble
        adjust = nxmit % 128
        if adjust != 0:
            nxmit += 128 - adjust

	# Make postample longer to align the packet of data on 128
    postamble = nxmit - (nsamples + preamble)
	#print "preamble", preamble, "postamble", postamble, "nsamples", nsamples, "nxmit", nxmit

	# Interleave preamble + samples + postable to transmit on I and Q
        # Scale so 0->1 maps to -tx_gain*(2^15-1)-> tx_gain*(2^15-1)
    tx_samples = numpy.zeros(2*nxmit,dtype=numpy.int16)

        # Make the preamble send the most negative value (a zero)
    tx_samples[0:2*preamble] = - tx_gain*32767.0

        # Now map the data into the right range and interleave
    tx_samples[2*preamble:2*preamble + 2*nsamples:2] = \
            (xmit.samples * 2.0  - 1.0) * tx_gain * 32767.0
    tx_samples[2*preamble + 1:2*preamble + 1 + 2*nsamples:2] = \
            (xmit.samples * 2.0 - 1.0) * tx_gain * 32767.0

        #Now do the postamble, again set to most negative value (aka a zero)
    tx_samples[2*preamble + 2*nsamples:2*preamble + 2*nsamples + 2*postamble:2] = - tx_gain*32767.0
    tx_samples[2*preamble + 2*nsamples+1:2*preamble + 2*nsamples + 2*postamble+1:2] = - tx_gain*32767.0

        ########
        # rx setup
        #######

        # decimation rate updated below once we know adc rate
    rx = usrp.usrp_rx(250)
    rx.stop()

        # set correct decimation rate
    decim = int((rx.adc_rate/sample_rate))
    assert decim <= 256, "usrp_ir_receive: sorry, sample rate must be at least %g" % rx.adc_rate/256
    rx.set_decim_rate(decim)
    decim_rate = rx.decim_rate

        # configure which ADCs to use
    rx.set_mux(0x10101010 if chan == 0 else 0x32323232)

        # set adc offset, disable input buffer since we're dc coupled
    if chan == 0:
        rx.set_adc_offset(0,rx_adc_offset)
        rx.set_adc_offset(1,rx_adc_offset)
        rx.set_adc_buffer_bypass(0,1)
        rx.set_adc_buffer_bypass(1,1)
    elif chan == 1:
        rx.set_adc_offset(2,rx_adc_offset)
        rx.set_adc_offset(3,rx_adc_offset)
        rx.set_adc_buffer_bypass(2,1)
        rx.set_adc_buffer_bypass(3,1)

        # disable DC offset removal control loop
    rx.set_dc_offset_cl_enable(0x0,0x3 if chan == 0 else 0xC)

        # set PGA gain
    assert rx_pga_gain_db >= rx.pga_min and rx_pga_gain_db <= rx.pga_max, "usrp_receive: rx_pga_gain_db must be in the range %g to %g" % (rx.pga_min,rx.pga_max)

        # DACs 0 and 1 share a gain setting, as doD ACs 2 and 3
    rx.set_pga(2*chan,rx_pga_gain_db)    # set it for both ADCs
    rx.set_pga(2*chan+1,rx_pga_gain_db)

        # set frequency of DUC
    assert rx.set_rx_freq(chan,frequency), "usrp_receive: set_rx_freq failed with value %g" % frequency

        # Make sure the number of active channels is one!!!!!
    assert rx.set_nchannels(1), "usrp_receive: set_nchannel failed with value 1" 

        # set up array to receive incoming samples
    nrcv = (nsamples + preamble + extra)/128 * 128   # buffer % 512 must = 0

    i_samples = numpy.zeros(nrcv,dtype=numpy.float32)
    q_samples = numpy.zeros(nrcv,dtype=numpy.float32)


        # Setup the transmitter thread
    tx_thread = tx_helper(tx,tx_samples)

        # Start the receiver
    assert rx.start(),"usrp receiver failed to start"

        # Start the transmitter thread
    tx_thread.start()

        # receive data loop
    count = 0
    datalist = numpy.zeros(1,dtype=numpy.int16)
    datasize = []
    while count < nrcv:
            # compute how many samples to read on this call
        want = min((nrcv-count)<<2,131072)

            # read 'em
        buffer,overrun = rx.read(want)
        if overrun == True:
            print("Buffer Overrun!!!")

            # deinterleave samples into I and Q buffers
            data = numpy.frombuffer(buffer,dtype=numpy.int16)
            got = data.size/2

            if got > 0:
                end = count + got
                i_samples[count:end] = data[0::2]
                q_samples[count:end] = data[1::2]
                count = end

        # done, stop receiver
        rx.stop()

        # wait for transmitter to complete in the background
        tx_thread.join()


        # Convert digital number to a value between -0.5 and 0.5
        numpy.multiply(i_samples,rx_gain/32768.0,i_samples)
        numpy.multiply(q_samples,rx_gain/32768.0,q_samples)

	# Offset the signed number so that data is always positive
        q_samples = q_samples + 0.5

	#print "preamble = ", preamble, "postamble = ", postamble
        # The MIT IR circuitry outputs to Q channel's ADC 
        return(sampled_waveform(q_samples[preamble-1:],sample_rate=sample_rate,domain='time'))

    def usrp_transmit(*waveforms,**keywords):
        gain = keywords.get('gain',1.0)
        board = keywords.get('board',None)
        dbid = keywords.get('dbid',usrp.USRP_DBID_LF_TX)
        pga_gain_db = keywords.get('pga_gain_db',0)
        frequency = keywords.get('frequency',0)
        repeat = keywords.get('repeat',True)

        nchan = len(waveforms)
        assert nchan >= 1 and nchan <= 2,"usrp_transmit: must supply one or two waveforms"
        nsamples = waveforms[0].nsamples
        sample_rate = waveforms[0].sample_rate
        domain = waveforms[0].domain
        for i in xrange(1,nchan):
            assert waveforms[i].nsamples==nsamples,\
                   "usrp_transmit: all waveforms must have the same number of samples"
            assert waveforms[i].sample_rate==sample_rate,\
                   "usrp_transmit: all waveforms must have the same sample rate"
            assert waveforms[i].domain==domain,\
                   "usrp_transmit: all waveforms must have the same domain"

        # adjust nsamples to be a multiple of 128 since we have to transmit
        # a multiple of 512 bytes and there are 4 bytes xmitted per sample
        adjust = nsamples % 128
        if adjust != 0:
            nsamples += 128 - adjust
            for i in xrange(nchan): waveforms[i].resize(nsamples)

        NBOARDS = 4   # how many boards to look for
        for b in xrange(NBOARDS):
            if not usrp.usrp_find_device(b): continue
            tx = usrp.usrp_tx(500,     # interp rate updated below
                              nchan = 1,
                              board = b)
            if board is None or board.lower() == tx.serial_number.lower():
                board = tx.serial_number
                break
            del tx
            tx = None
        else:
            assert False,"usrp_transmit: can't find USRP board %s" % board

        # set correct interpolation rate
        interp = int(tx.dac_rate/sample_rate)
        assert interp <= 512,\
               "usrp_transmit: sorry, sample rate must be at least %g" % tx.dac_rate/512
        tx.set_interp_rate(interp)
        interp_rate = tx.interp_rate

        # send data to correct daughterboard
        if dbid == 0 or tx.daughterboard_id(0) == dbid:
            chan = 0
            tx.set_mux(0x0098)  # send chan 0 to DACs 0 and 1
        elif dbid == 1 or tx.daughterboard_id(1) == dbid:
            chan = 1
            tx.set_mux(0x9800)  # send chan 0 to DACs 2 and 3
        else:
            assert False,"usrp_transmit: can't find daughterboard with id 0x%x" % dbid

        # set PGA gain
        assert pga_gain_db >= tx.pga_min and pga_gain_db <= tx.pga_max,\
               "usrp_transmit: pga_gain_db must be in the range %g to %g" % (tx.pga_min,tx.pga_max)
        # DACs 0 and 1 share a gain setting, as do DACs 2 and 3
        tx.set_pga(chan << 1,pga_gain_db)

        # set frequency of DUC
        assert tx.set_tx_freq(chan,frequency),\
               "usrp_transmit: set_tx_freq failed with value %g" % frequency

        # scale and interleave the waveform samples
        nxmit = 2*nsamples
        samples = numpy.zeros(nxmit,dtype=numpy.int16)
        for i in xrange(nchan):
            samples[i::2] = numpy.multiply(waveforms[i].samples,gain*32767.0)

        tx.stop()
        assert tx.start(),"usrp_transmit: usrp failed to start"

        # try sending what we have
        while True:
            offset = 0   # where we are in transmitting data
            while offset < nxmit:
                end = min(offset+131072,nxmit)
                data = samples[offset:end].tostring()
                bytes_sent,underrun = tx.write(data)
                if underrun: print('usrp_transmit: underrun...')
                offset += bytes_sent/2
            if not repeat: break

        tx.wait_for_completion()
        tx.stop()

    def usrp_receive(sample_rate,nsamples,
                     board=None,dbid=usrp.USRP_DBID_LF_RX,pga_gain_db=0,
                     frequency=0,gain=1.0,raw=False):

        NBOARDS = 4   # how many boards to look for
        for b in xrange(NBOARDS):
            if not usrp.usrp_find_device(b): continue
            rx = usrp.usrp_rx(250,     # interp rate updated below
                              nchan = 1,
                              board = b)
            if board is None or board.lower() == rx.serial_number.lower():
                board = rx.serial_number
                break
            del rx
            rx = None
        else:
            assert False,"usrp_receive: can't find USRP board %s" % board

        # set correct decimation rate
        decim = int(rx.adc_rate/sample_rate)
        assert decim <= 256,\
               "usrp_receive: sorry, sample rate must be at least %g" % rx.adc_rate/256
        rx.set_decim_rate(decim)
        decim_rate = rx.decim_rate

    rx_adc_offset = 0

        # get data from correct daughterboard
    if dbid == 0 or rx.daughterboard_id(0) == dbid:
        chan = 0
        rx.set_dc_offset_cl_enable(0x0,0x3)
        rx.set_mux(0x10101010)  # use first pair of ADCs
    elif dbid == 1 or rx.daughterboard_id(1) == dbid:
        chan = 1
        rx.set_dc_offset_cl_enable(0x0,0xC)
        rx.set_mux(0x32323232)  # use second pait of ADCs
        rx.set_adc_offset(2,rx_adc_offset)
        rx.set_adc_offset(3,rx_adc_offset)
        rx.set_adc_buffer_bypass(2,1)
        rx.set_adc_buffer_bypass(3,1)
    else:
        assert False,"usrp_receive: can't find daughterboard with id 0x%x" % dbid

        # set PGA gain
    assert pga_gain_db >= rx.pga_min and pga_gain_db <= rx.pga_max,\
               "usrp_receive: pga_gain_db must be in the range %g to %g" % (rx.pga_min,rx.pga_max)
        # DACs 0 and 1 share a gain setting, as do DACs 2 and 3
    rx.set_pga(2*chan,pga_gain_db)    # set it for both ADCs
    rx.set_pga(2*chan+1,pga_gain_db)

        # set frequency of DUC
    assert rx.set_rx_freq(chan,frequency),\
               "usrp_receive: set_rx_freq failed with value %g" % frequency

        # scale and interleave the waveform samples
    isamples = numpy.zeros(nsamples,dtype=numpy.float32)
    qsamples = numpy.zeros(nsamples,dtype=numpy.float32)

    rx.stop()

    assert rx.start(),"usrp_receive: usrp failed to start"

    count = 0
    while count < nsamples:
        want = min((nsamples-count)<<2,131072)
        buffer,overrun = rx.read(want)
        data = numpy.frombuffer(buffer,dtype=numpy.int16)
        got = data.size/2
        if got > 0:
            end = count + got
            isamples[count:end] = data[0::2]
            qsamples[count:end] = data[1::2]
            count = end

    rx.stop()
    if not raw:
        numpy.multiply(isamples,gain/32768.0,isamples)
        numpy.multiply(qsamples,gain/32768.0,qsamples)

    return(sampled_waveform(isamples,sample_rate=sample_rate,domain='time'),
               sampled_waveform(qsamples,sample_rate=sample_rate,domain='time'))

except ImportError:
    def usrp_transmit(*waveforms,**keywords):
        raise: NotImplementedError, 'sorry, no usrp library is available"

    def usrp_receive(*waveforms,**keywords):
        raise NotImplementedError:"sorry, no usrp library is available"

################################################################################
## operations
################################################################################

def phase_angle(x,y):
    ysamples = x._check(y)
    return sampled_waveform(numpy.arctan2(x.samples,ysamples),
                            sample_rate=x.sample_rate)

def phase_difference(phases):
    result = numpy.convolve(phases.samples,[1,-1],mode='same')
    result[result > numpy.pi] -= 2*numpy.pi
    result[result < -numpy.pi] += 2*numpy.pi
    result[0] = 0
    return sampled_waveform(result,sample_rate=phases.sample_rate)


def symbols_to_samples(symbols,samples_per_symbol,sample_rate,sample_values=None):
    nsymbols = len(symbols)
    if sample_values is None:
        sample_values = numpy.arange(nsymbols)
    samples = numpy.zeros(nsymbols*samples_per_symbol,dtype=numpy.int)
    for j in xrange(samples_per_symbol):
        samples[j::samples_per_symbol] = sample_values[symbols]
    return sampled_waveform(samples,sample_rate=sample_rate)

def samples_to_centers(samples,samples_per_symbol):
    nsamples = len(samples)

    # try to find a good threshold for digitizing the samples
    threshold = numpy.average(samples.samples)

    # digitize the samples
    dsamples = numpy.zeros(nsamples,dtype=numpy.int)
    dsamples[samples.samples >= threshold] = 1

    # now use a little control loop to figure out where to sample
    centers = numpy.zeros(nsamples/samples_per_symbol,dtype=numpy.int)
    sample_point = samples_per_symbol/2
    sample_end = samples_per_symbol - 1
    index = 0
    count = 0
    last = 0
    for i in xrange(nsamples):
        current = dsamples[i]
        if count == sample_end or last != current:
            count = 0
            last = current
        else:
            if count == sample_point and index < centers.size:
                centers[index] = i
                index += 1
            count = count + 1

    centers.resize(index)

    """
    p.figure()
    p.plot(samples.samples)
    p.plot(dsamples)
    p.plot(centers,[0.5]*len(centers),'ro')
    p.show()
    sys.exit()
    """

    return centers

################################################################################
## testing code
################################################################################

if __name__=='__main__':
    pass
