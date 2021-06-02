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
import warnings

warnings.filterwarnings("ignore")


import librosa

model=load_model('C:/collegework/CDSAML/training/best_accuracy/weights-improvement-833-0.97.hdf5')
adam=keras.optimizers.Adam(lr=0.0001,beta_1=0.9,beta_2=0.999, epsilon=None,decay=0.0,amsgrad=False)
model.compile(loss='categorical_crossentropy',optimizer=adam,metrics=['accuracy'])

def detect_leading_silence(sound,silence_threshold=-35.0,chunk_size=40):
    split_audio=split_on_silence(sound,min_silence_len=140,silence_thresh=-16,keep_silence=50
                                 )
    return split_audio

def Predict_audio(data_path):
    audio_data_list=[]
    num_channel=1
    X,sample_rate = librosa.load(data_path,res_type='kaiser_fast',mono=False)
    #X=librosa.to_mono(X)
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
    op=Predict_audio(data_path)
    double=0
    triple=0
    hundred=0
    thousand=0
    op_str=''
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

    time.sleep(.5)
    return op_str[:len(op_str)-1]
    #print(op_str)
    
    

def path():
    data_path="C:/collegework/CDSAML/audio/arjunone1.wav"
    if os.path.exists(data_path):
       l=detect(data_path)    
       print(l)
       #os.remove(data_path)
    time.sleep(1)
        
path() #this will trigger everything

"""
chunk 0(double)

chunk 1(2)
chunk 3(3)
chunk 6(9)
chunk 8
chunk 15(9)
chunk 12(2)
chunk 11(3)
chunk 12(2)
"""

