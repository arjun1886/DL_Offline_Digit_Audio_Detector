
# coding: utf-8

# In[84]:


from scipy.io import wavfile
import numpy as np
import pandas as pd
from scipy import signal
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelBinarizer

import keras
from keras import backend as K
from keras.layers.core import Dense,Dropout,Activation,Flatten
from keras.layers.convolutional import Convolution1D
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.optimizers import SGD,RMSprop,adam
from keras.models import Model, Sequential, load_model
from keras.optimizers import Adam
from keras.callbacks import TensorBoard
from keras.utils import np_utils


# In[85]:


from __future__ import print_function
import numpy as np
import librosa
from random import getrandbits
import sys, getopt, os
#from scipy.signal import resample     # too slow


def random_onoff():                # randomly turns on or off
    return bool(getrandbits(1))


# returns a list of augmented audio data, stereo or mono
def augment_data(y, sr, n_augment = 0, allow_speedandpitch = False, allow_pitch = False,
    allow_speed = False, allow_dyn = False, allow_noise = True, allow_timeshift = False, tab=""):

    mods = [y]                  # always returns the original as element zero
    length = y.shape[0]

    for i in range(n_augment):
#         print(tab+"augment_data: ",i+1,"of",n_augment)
        y_mod = y
        count_changes = 0

        # change speed and pitch together
        if (allow_speedandpitch) and random_onoff():   
            length_change = np.random.uniform(low=0.9,high=1.1)
            speed_fac = 1.0  / length_change
#             print(tab+"    resample length_change = ",length_change)
            tmp = np.interp(np.arange(0,len(y),speed_fac),np.arange(0,len(y)),y)
            #tmp = resample(y,int(length*lengt_fac))    # signal.resample is too slow
            minlen = min( y.shape[0], tmp.shape[0])     # keep same length as original; 
            y_mod *= 0                                    # pad with zeros 
            y_mod[0:minlen] = tmp[0:minlen]
            count_changes += 1

        # change pitch (w/o speed)
        if (allow_pitch) and random_onoff():   
            bins_per_octave = 24        # pitch increments are quarter-steps
            pitch_pm = 4                                # +/- this many quarter steps
            pitch_change =  pitch_pm * 2*(np.random.uniform()-0.5)   
#             print(tab+"    pitch_change = ",pitch_change)
            y_mod = librosa.effects.pitch_shift(y, sr, n_steps=pitch_change, bins_per_octave=bins_per_octave)
            count_changes += 1

        # change speed (w/o pitch), 
        if (allow_speed) and random_onoff():   
            speed_change = np.random.uniform(low=0.9,high=1.1)
#             print(tab+"    speed_change = ",speed_change)
            tmp = librosa.effects.time_stretch(y_mod, speed_change)
            minlen = min( y.shape[0], tmp.shape[0])        # keep same length as original; 
            y_mod *= 0                                    # pad with zeros 
            y_mod[0:minlen] = tmp[0:minlen]
            count_changes += 1

        # change dynamic range
        if (allow_dyn) and random_onoff():  
            dyn_change = np.random.uniform(low=0.5,high=1.1)  # change amplitude
#             print(tab+"    dyn_change = ",dyn_change)
            y_mod = y_mod * dyn_change
            count_changes += 1

        # add noise
        if (allow_noise) and random_onoff():  
            noise_amp = 0.05*np.random.uniform()*np.amax(y)  
            if random_onoff():
#                 print(tab+"    gaussian noise_amp = ",noise_amp)
                y_mod +=  noise_amp * np.random.normal(size=length)  
            else:
#                 print(tab+"    uniform noise_amp = ",noise_amp)
                y_mod +=  noise_amp * np.random.normal(size=length)  
            count_changes += 1

        # shift in time forwards or backwards
        if (allow_timeshift) and random_onoff():
            timeshift_fac = 0.2 *2*(np.random.uniform()-0.5)  # up to 20% of length
#             print(tab+"    timeshift_fac = ",timeshift_fac)
            start = int(length * timeshift_fac)
            if (start > 0):
                y_mod = np.pad(y_mod,(start,0),mode='constant')[0:y_mod.shape[0]]
            else:
                y_mod = np.pad(y_mod,(0,-start),mode='constant')[0:y_mod.shape[0]]
            count_changes += 1

        # last-ditch effort to make sure we made a change (recursive/sloppy, but...works)
        if (0 == count_changes):
#             print("No changes made to signal, trying again")
            mods.append(  augment_data(y, sr, n_augment = 1, tab="      ")[1] )
        else:
            mods.append(y_mod)

    return mods


# In[86]:



# # Load the example track
# y, sr = librosa.load("sampleAudio/hello.ogg")
# # ipd.Audio(y, rate=sr)

# modded = augment_data(y, sr,1)
# print()
# ipd.Audio(modded[1],rate=sr)


# In[87]:


# modded = np.array(modded)
# print(modded.shape)
# modded = np.sum(modded, axis=0)
# print(modded.shape)

# ipd.Audio(modded, rate = sr)


# In[88]:



def get_model(num_classes, shape):
    
    '''Create a keras model.'''
    
    model = Sequential()
    model.add(Dense(60, input_shape=shape, name="input"))
    model.add(Dropout(0.33))
    model.add(Activation('relu'))
    
    
    model.add(Dense(60))
    model.add(Dropout(0.33))
    model.add(Activation('relu'))
    
    model.add(Flatten())
    model.add(Dense(40))
    model.add(Activation('relu'))
    model.add(Dropout(0.33))
    model.add(Dense(14,activation='softmax',name='op'))
    
    
    adam=keras.optimizers.Adam(lr=0.01)
    model.compile(loss='categorical_crossentropy', optimizer= adam, metrics=['accuracy'])
    
    return model
    
    

def save_model(model, model_path):
    # save model
    model_json = model.to_json()
    with open(os.path.join(os.path.abspath(model_path), 'model.json'), 'w') as json_file:
        json_file.write(model_json)
        

        
def get_callbacks(model_path):
    
    # save weights of best training epoch: monitor either val_loss or val_acc

    weights_path = os.path.join(os.path.abspath(model_path), 'weights-improvement-{epoch:02d}-{val_acc:.2f}.h5')
    callbacks_list = [
        ModelCheckpoint(weights_path, monitor='val_acc', verbose=1, save_best_only=True),
        EarlyStopping(monitor='val_acc', patience=10, verbose=0),
        keras.callbacks.TensorBoard(log_dir='tensorboard/training', histogram_freq=0, write_graph=False, write_images=False)
    ]
    
    return callbacks_list    

def batch_generator(X, y, batch_size=64, sample_rate=16000):
    '''
    Return a random image from X, y
    '''
    
    while True:
        # choose batch_size random images / labels from the data
        idx = np.random.randint(0, X.shape[0], batch_size)
        im = X[idx]
        label = y[idx]
        
        
        modded = []
        
        for i in im:
#             print("Shape of I" , i.shape)
            augmented_data = augment_data(i, sample_rate, 1)[1]
#             augmented_data = i
            mfccs = np.median(librosa.feature.mfcc(y=augmented_data,sr=sample_rate, n_mfcc=64),axis=1)
            mfccs = mfccs.reshape((mfccs.shape[0],1))
#             print("Shape of one example ", mfccs.shape)
            modded.append(mfccs)
        
        audio_data = np.array(modded)
        audio_data = audio_data.astype('float32')
        audio_data /= 255.
#         print("Shape of batch", audio_data.shape)
        label = label.reshape((batch_size, label.shape[1]))
#         print("Shape of label", label.shape)
        
        yield audio_data, label


# In[89]:


def get_class_mappings(directory):
    class_mappings = dict()
    for index, value in enumerate(os.listdir(directory)):
        class_mappings[value] = index

    return class_mappings


def get_data(directory, nsamples=16000, num_classes=14):
    
    class_mappings = get_class_mappings(directory)
    
    audio_data = []
    labels = []
    for digit_name in os.listdir(directory):
        for file_name in os.listdir(os.path.join(directory, digit_name)):
            
            X, sample_rate = librosa.load(os.path.join(directory, digit_name, file_name), res_type='kaiser_fast')
#             if X.size < 20000:
#                 X = np.pad(X, (nsamples - X.size, 0), mode='constant')
#             else:
#                 X = X[0:nsamples]
                
            audio_data.append(X)
            
            labels.append(class_mappings.get(digit_name))

    audio_data = np.array(audio_data)
    labels = np.array(labels)
    
    return audio_data,np_utils.to_categorical(labels, num_classes=num_classes), sample_rate

    
   
def main():    
    
    batch_size = 64
    num_classes = 14
    shape = (64,1)
    DATA_DIR = "./data"
    MODEL_DIR = "./weights"
    
    print("[INFO] Getting Data")
    X, Y, sr = get_data(DATA_DIR)
#     X, Y = shuffle(X,Y, random_state=4)
    
    print("[INFO] Splitting Data")
    #Split the dataset
    X_train ,X_test, y_train, y_test = train_test_split(X, Y, test_size=0.15, random_state=3) 
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.20, random_state=3)
    
#     shape = X_train[0].shape
    
    print("[INFO] Getting Model")
    model = get_model(num_classes, shape)
    callbacks = get_callbacks(MODEL_DIR)
    
    print("[INFO] Creating Generators")
    train_generator = batch_generator(X_train, y_train,batch_size, sr)
    validation_generator = batch_generator(X_val, y_val,batch_size, sr)
    test_generator = batch_generator(X_test, y_test,batch_size, sr)

    print("[INFO] Starting to Train")
    model.fit_generator(generator=train_generator,
                        epochs=1000,
                        verbose = 1,
                        steps_per_epoch=X_train.shape[0] // 32,
                        validation_data=validation_generator,
                        validation_steps=X_val.shape[0] // 32,
                        callbacks=callbacks)
    
    
    save_model(model, MODEL_DIR)
    


# In[90]:


# X, Y, sr = get_data("./data")
main()


# In[ ]:



# gen = batch_generator(X, Y, 64,sr)

# for x,y in gen:
#     print(x.shape)
#     print(y.shape)
#     break
    
    

