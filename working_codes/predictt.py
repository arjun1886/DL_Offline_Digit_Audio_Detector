import pyaudio
import wave
import pydub
import itertools
import random
from pydub import AudioSegment
import time
from pydub.silence import split_on_silence
import keras
import os
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from keras import backend as K
from keras.utils import np_utils
from keras.models import Sequential
from keras.optimizers import SGD,RMSprop,adam
from keras.models import load_model
import numpy as np

import librosa

import warnings
warnings.filterwarnings("ignore")

model=load_model('C:/collegework/CDSAML/training/best_accuracy/weights-improvement-1953-0.98.hdf5')
adam=keras.optimizers.Adam(lr=0.0001,beta_1=0.9,beta_2=0.999, epsilon=None,decay=0.0,amsgrad=False)
model.compile(loss='categorical_crossentropy',optimizer=adam,metrics=['accuracy'])

def detect_leading_silence(sound,silence_threshold=-35.0,chunk_size=40):
    split_audio=split_on_silence(sound,min_silence_len=200,silence_thresh=-35,keep_silence=200
                                 )
    return split_audio

def Predict_audio(data_path):
    audio_data_list=[]
    num_channel=1
    X,sample_rate = librosa.load(data_path,res_type='kaiser_fast',mono=False)
    X=librosa.to_mono(X)
    mfccs = np.mean(librosa.feature.mfcc(y=X,sr=sample_rate, n_mfcc=60).T,axis=0)
    
    audio_data_list.append(mfccs)
    audio_data = np.array(audio_data_list)
    audio_data = audio_data.astype('float32')
    audio_data/=255
    #print("audio data.shape")
    
    if num_channel==1:
       if K.image_dim_ordering()=='th':
          audio_data= np.expand_dims(audio_data,axis=1)
          #print (audio data.shape)
       else:
          audio_data=np.expand_dims(audio_data,axis=4)
          #print (audio data.shapel
    
       test_image = audio_data
       #print (test_image.shape)
    
       (model.predict(test_image))
       return ((model.predict_classes(test_image))[0])

def detect(data_path):
    sound = AudioSegment.from_file(data_path,format=".wav")
    #sound.apply_gain(5)
    #split_audio = split_on_silence(sound,min_silence_len=10,silence_thresh=-28, keep_silence=20)
    split_audio=detect_leading_silence(sound)
    c=0
    op_str='' #op string to be returned
    double=0
    triple=0
    hundred=0
    thousand=0
    for i in range(len(split_audio)):
        trimmed_sound=split_audio[i]
        if(len(trimmed_sound)>0):
            trimmed_sound.export("C:/collegework/CDSAML/audio/"+str(i)+".wav",format='wav')
            
    for(dirpath, subdir, filenames) in os.walk("C:/collegework/CDSAML/audio/"):   #review this line
       for fl in filenames:
            sound="C:/collegework/CDSAML/audio/"+fl     
            op=Predict_audio(sound)
            if(int(op)==10):
               double=double+1
            elif (int(op)==11):
                 triple=triple+1
            elif (int(op)==12):
                 hundred+=1
                 for i in range(hundred*2):
                     op_str=op_str+'0'+'-'
                 hundred=0
            elif (int(op)==13):
                 thousand+=1
                 for i in range(thousand*3):
                     op_str=op_str+'0'+'-'
                 thousand=0
            elif(((double>0) and (triple==0)) or ((triple>0) and (double==0))):
                 if double>0:
                    for i in range(double*2):
                        op_str=op_str+str(op)+'-'
                    double=0
                
                
                 else:
                    for i in range(triple*3):
                        op_str=op_str+str(op)+'-'
                    triple=0
            else:
                op_str=op_str+str(op)+'-'
                
                
            #os.remove("C:/collegework/CDSAML/audio/"+str(c) + '.wav')
        
        
    return op_str
    #print(op_str)
    
    
def path():
    
    FORMAT = pyaudio.paInt16
    CHANNELS = 1
    RATE = 16000
    CHUNK = 1024
    # RECORD_SECONDS = int(input("enter no of seconds you want to record audio"))
    RECORD_SECONDS = 9
    
    for j in range(1): #specifyhow many times you want to record 0-9,double,triple..
        
        print("\n"+str(j-1))
        for i in range(1):
            print("speak out:")
            time.sleep(1)
            data_path="C:/collegework/CDSAML/file.wav" #put your file in this path
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
            
            waveFile=wave.open(data_path,'wb')
            waveFile.setnchannels(CHANNELS)
            waveFile.setsampwidth(audio.get_sample_size(FORMAT))
            waveFile.setframerate(RATE)
            waveFile.writeframes(b''.join(frames))
            waveFile.close()
    
    if os.path.exists(data_path):
       l=detect(data_path)    
       print(l)
    time.sleep(1)
        
path() #this will trigger everything





