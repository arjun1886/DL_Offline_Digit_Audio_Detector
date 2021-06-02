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
RECORD_SECONDS = 3
val = 9
num_display=[]


numbers = ['shoonya','ek','dho','theen','char','paanch','cheh','saath','aat','nau']
#numbers=['hundred']
for j in range(3): #specifyhow many times you want to record 0-9,double,triple.
    a=random.randint(1,1000)
    print("\n"+str(j-1))
    val+=1
    for i in range(10):
        print("speak out:" + numbers[i] )
        
        time.sleep(1)
        WAVE_OUTPUT_FILENAME="C:/collegework/CDSAML/data_hindi/"+numbers[i]+"/"+"arjun"+numbers[i]+str(val)+".wav"
        audio= pyaudio.PyAudio()
        
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