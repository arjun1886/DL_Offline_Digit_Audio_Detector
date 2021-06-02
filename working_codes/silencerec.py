import pyaudio
import random
import wave
import time
from pydub import AudioSegment
from pydub.playback import play
from random import shuffle

FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000
CHUNK = 1024
#RECORD_SECONDS = int(input("enter no of seconds you want to record audio"))

val = 1
num_display=[]
RECORD_SECONDS = 0.4


numbers = ['silencee']  
for i in range(80):
    print("speak out:" + numbers[0] )
    val+=1
    time.sleep(1)
    WAVE_OUTPUT_FILENAME="C:/collegework/CDSAML/trimmed_englishh/silence/"+numbers[0]+str(val)+".wav"
    audio= pyaudio.PyAudio()
    RECORD_SECONDS +=0.015
    #start recording
    stream=audio.open(format=FORMAT, channels=CHANNELS,rate=RATE,input=True,output=True,frames_per_buffer=CHUNK)
    print("recording..")
    frames=[]
    
    for i in range(0,int(RATE/CHUNK*RECORD_SECONDS)):
        data= stream.read(CHUNK)
        frames.append(data)
    #finished recording
    
    #stop recording
    stream.stop_stream()
    stream.close()
    audio.terminate()
    
    waveFile=wave.open(WAVE_OUTPUT_FILENAME,'wb')
    waveFile.setnchannels(CHANNELS)
    waveFile.setsampwidth(audio.get_sample_size(FORMAT))
    waveFile.setframerate(RATE)
    waveFile.writeframes(b''.join(frames))
    waveFile.close()