import pyaudio
import numpy as np 
import matplotlib.pyplot as plt
import wave
import sys
import scipy.io.wavfile as WaveFile
import scipy.signal as pysignal 
'''
import wave
import struct

def pcm_channels(wave_file):
    """Given a file-like object or file path representing a wave file,
    decompose it into its constituent PCM data streams.

    Input: A file like object or file path
    Output: A list of lists of integers representing the PCM coded data stream channels
        and the sample rate of the channels (mixed rate channels not supported)
    """
    stream = wave.open(wave_file,"rb")

    num_channels = stream.getnchannels()
    sample_rate = stream.getframerate()
    sample_width = stream.getsampwidth()
    num_frames = stream.getnframes()

    raw_data = stream.readframes( num_frames ) # Returns byte data
    stream.close()

    total_samples = num_frames * num_channels

    if sample_width == 1: 
        fmt = "%iB" % total_samples # read unsigned chars
    elif sample_width == 2:
        fmt = "%ih" % total_samples # read signed 2 byte shorts
    else:
        raise ValueError("Only supports 8 and 16 bit audio formats.")

    integer_data = struct.unpack(fmt, raw_data)
    del raw_data # Keep memory tidy (who knows how big it might be)

    channels = [ [] for time in range(num_channels) ]

    for index, value in enumerate(integer_data):
        bucket = index % num_channels
        channels[bucket].append(value)

    return channels, sample_rate
'''
def checkcompatability(filename1,filename2):
	wf1=wave.open(filename1,'rb')
	wf2=wave.open(filename2,'rb')
	if (wf1.getsampwidth()==wf2.getsampwidth()) and (wf1.getnchannels()==wf2.getnchannels()) and (wf1.getframerate()==wf2.getframerate()):
		return [wf1.getsampwidth(),wf1.getnchannels(),wf1.getframerate()]
	else:
		return [None,None,None]

def plotx(x,n,title):
	plt.figure(n)
	plt.plot(x)
	plt.title(title)

def plotxy(x,y,n,title):
	plt.figure(n)
	plt.plot(x,y)
	plt.title(title)	

def audiodata(filename):
	CHUNK=1024
	wf=wave.open(filename,'rb')
	frame=wf.readframes(CHUNK)
	data=[]
	while len(frame)>0:
		data.append(frame)
		frame=wf.readframes(CHUNK)
	data=''.join(data)
	audiodata=np.fromstring(data,np.int16)
	wf.close()
	return audiodata

def fftshift(audiodata):
	fft=np.fft.fft(audiodata)
	fftshift=np.fft.fftshift(fft)
	return np.real(fftshift)

def inversefft(data):
	ifshift=np.fft.ifftshift(data)
	ifft=np.fft.ifft(ifshift)
	return np.real(ifft)

def savefile(filename,audiodata,rate):
	data=audiodata.astype(np.int16)
	WaveFile.write(filename+".wav",rate,data)
def savefiltered(filename,audiodata,rate):
	data=audiodata.astype(np.float32)
	WaveFile.write(filename+".wav",rate,data)
def echofilter(data,technique,R,a,N=None):
	if(technique=="echofilter"):
		num=np.concatenate((np.array([1]),np.zeros(R-1),np.array([a])),axis=0)
		den=np.array([1])
	elif(technique=="finiteechoing"):
		num=np.concatenate((np.array([1]),np.zeros(R*N-1),np.array([-a**N])),axis=0)
		den=np.concatenate((np.array([1]),np.zeros(R-1),np.array([-a])),axis=0)
	elif(technique=="infiniteechoing"):
		num=np.concatenate((np.array([0]),np.zeros(R),np.array([1])),axis=0)
		den=np.concatenate((np.array([1]),np.zeros(R),np.array([-a])),axis=0)
	else:
		num=np.concatenate((np.array([a]),np.zeros(R-1),np.array([1])),axis=0)
		den=np.concatenate((np.array([1]),np.zeros(R-1),np.array([a])),axis=0)

	out=pysignal.lfilter(num,den,data)
	[w,h]=pysignal.freqz(num,den)
	plotxy(w/np.pi,20*np.log10(np.abs(h)),1,"Magnitude response of filter Normalized frequecy vs magnitude")			
	normout=(2*((out-np.min(out))/(np.max(out)-np.min(out))))-1;	
	return normout
if __name__=="__main__":
	if(len(sys.argv)==2):
		choice=sys.argv[1]
		if choice=="Mix":
			file1name=raw_input("Enter audio file 1 with full name(Including extension) :")
			file2name=raw_input("Enter audio file 1 with full name(Including extension) :")
			samplewidth,channels,rate=checkcompatability(file1name,file2name)
			print samplewidth,channels,rate
			if (samplewidth!=None) and (channels!=None) and (rate!=None):
				file1data=audiodata(file1name)
				file2data=audiodata(file2name)
			
				plotx(file1data,1,"Audio file 1 time domain plot")
				plotx(file2data,2,"Audio file 2 time domain plot")
			
				fshift1=fftshift(file1data)
				fshift2=fftshift(file2data)

				f=np.arange(-rate/2,rate/2,rate/float(len(file1data)))
			
				plotxy(f,fshift1,3,"Audio file 1 frequency domain plot")
				plotxy(f,fshift2,4,"Audio file 2 frequency domain plot")
			
				fshiftout=fshift2+fshift1
				plotxy(f,fshiftout,5,"Ouput file frequency domain plot")
			
				output=inversefft(fshiftout)
			
				savefile("output",output,rate)
				plt.show()
			else:
				if(samplewidth==None):
					print "The sample width of two files are not equal"
				if(channels==None):
					print "The channel of two files are not equal"
				if(rate==None):
					print "The rate of two file are not equal"
				sys.exit()
		elif choice=="Filter":
			c=int(raw_input("Types of filters\n1.Echo filter\n2.finite multiple Echo filter\n4.infinite multiple Echo filter\n4.All pass reverberator\n5.Exit\nEnter you Filtering choice:"))
			if(c==5):
				sys.exit()
			filename=raw_input("Enter file name with full file extension:");
			filedata=audiodata(filename)
			R=int(raw_input("Enter the feedback frequency:"))
			a=float(raw_input("Enter the feedback magnitude:"))
			if(c==1):
				filtereddata=echofilter(filedata,"echofilter",R,a)
				plotx(filtereddata,2,"Echo Filtered audio data")
				savefiltered("echodoutput_r"+str(R)+"_a"+str(a),filtereddata,44100)
			elif(c==2):
				N=int(raw_input("Enter the no of echo's:"))
				filtereddata=echofilter(filedata,"finiteechoing",R,a,N)
				plotx(filtereddata,2,"Finite Multiple Echo Filtered audio data")
				savefiltered("finiteechoing_r"+str(R)+"_a"+str(a)+"_N"+str(N),filtereddata,44100)
			elif(c==3):
				filtereddata=echofilter(filedata,"infiniteechoing",R,a)
				plotx(filtereddata,2,"Infinite Multiple Echo Filtered audio data")
				savefiltered("infiniteechoing_r"+str(R)+"_a"+str(a),filtereddata,44100)
			elif(c==4):
				filtereddata=echofilter(filedata,"reverb",R,a)
				plotx(filtereddata,2,"Reverb Filtered audio data")
				savefiltered("reverb_r"+str(R)+"_a"+str(a),filtereddata,44100)
			plt.show()
		else:
			print "Input error given argument not valid Ex: python mix.py Mix or Filter"
	else:
		print "please specify operation to perform Ex: python mix.py Mix or Filter"

	
	
	
