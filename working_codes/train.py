import os
import numpy as np

from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
#from sklearn.cross validation import train_test_split
from keras import backend as K
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers.core import Dense,Dropout,Activation,Flatten
from keras.layers.convolutional import Convolution1D
from keras.optimizers import SGD,RMSprop,adam
from keras.models import load_model
import keras
from keras.callbacks import ModelCheckpoint
import librosa

#Define data path
data_path = "C:/collegework/CDSAML/trimmed_all/"
data_dir_list = os.listdir(data_path)

map=[[1 for i in range(2)] for j in range(14)]

for i in range(len(data_dir_list)):
    if data_dir_list[i]=='double':
       map[i][0]='double'
       map[i][1]=10
    elif data_dir_list[i]=='eight':
       map[i][0]='eight'
       map[i][1]=8
    elif data_dir_list[i]=='five':
       map[i][0]='five'
       map[i][1]=5
    elif data_dir_list[i]=='four':
       map[i][0]='four'
       map[i][1]=4
    elif data_dir_list[i]=='nine':
       map[i][0]='nine'
       map[i][1]=9
    elif data_dir_list[i]=='one':
       map[i][0]='one'
       map[i][1]=1
    elif data_dir_list[i]=='seven':
       map[i][0]='seven'
       map[i][1]=7
    elif data_dir_list[i]=='six':
       map[i][0]='six'
       map[i][1]=6
    elif data_dir_list[i]=='three':
       map[i][0]='three'
       map[i][1]=3
    elif data_dir_list[i]=='triple':
       map[i][0]='triple'
       map[i][1]=11
    elif data_dir_list[i]=='two':
       map[i][0]='two'
       map[i][1]=2
    elif data_dir_list[i]=='hundred':
       map[i][0]='hundred'
       map[i][1]=12
    elif data_dir_list[i]=='thousand':
       map[i][0]='thousand'
       map[i][1]=13
    else:
       map[i][0]='zero'
       map[i][1]=0
num_channel=1

#Define the number of classes
num_classes = 14

audio_data_list=[]
size_data=[]
lengths=[]


for dataset in data_dir_list:
    audio_list=os.listdir(data_path+'/'+ dataset)
    "print('Loaded the images of dataset-'+'{}\n'.format(dataset))"
    lengths.append(len(audio_list))
        
        
    for audio in audio_list:
        X, sample_rate = librosa.load(data_path+'/'+ dataset +'/'+audio, res_type='kaiser_fast')
        #we extract mfcc feature from data
        mfccs = np.mean(librosa.feature.mfcc(y=X,sr=sample_rate, n_mfcc=60).T,axis=0)

        audio_data_list.append(mfccs)
size_data.append(len(audio_list))

#print(data_dir_list)
audio_data = np.array(audio_data_list)
audio_data = audio_data.astype('float32')
audio_data /= 255
#print(audio_data[0])

#print(audio_data.shape)

if num_channel==1:
	if K.image_dim_ordering()=='th':
		audio_data=np.expand_dims(audio_data, axis=1)
		print(audio_data.shape)

	else: 
		audio_data=np.expand_dims(audio_data, axis=4)
		print(audio_data.shape)

num_classes = 14

num_of_samples = audio_data.shape[0]
#print(num_of_samples)
labels = np.ones((num_of_samples,),dtype='int64')

i=1
while(i<14):
     lengths[i]=lengths[i]+lengths[i-1]
     i+=1

print(lengths)

for k in range(0,lengths[0]):
    labels[k]=map[0][1]


i=1
while(i<14):
     for j in range(lengths[i-1],lengths[i]):
         labels[j]=map[i][1]
     i+=1


names =[]
for i in data_dir_list:
	names.append(i)

Y= np_utils.to_categorical(labels,num_classes)
x,y = shuffle(audio_data,Y, random_state=4)

#Split the dataset
X_train,X_test, y_train, y_test = train_test_split(x, y, test_size=0.15, random_state=3) 
X_train,X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.20, random_state=3)

#Defining the model
shape=X_train[0].shape
print(shape)

model = Sequential()
model.add(Dense(40, input_shape=shape, name="input"))
model.add(Dropout(0.33))
model.add(Activation('relu'))


model.add(Dense(40))
model.add(Dropout(0.33))
model.add(Activation('relu'))

model.add(Flatten())
model.add(Dense(30))
model.add(Activation('relu'))
model.add(Dropout(0.33))
model.add(Dense(num_classes,activation='softmax',name='op'))

adam=keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss='categorical_crossentropy', optimizer= adam, metrics=['accuracy'])

filepath = 'C:/collegework/CDSAML/training/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='val_acc',verbose=1,save_best_only=True,mode='max')
callback_list = [checkpoint]
model.fit(X_train,y_train, epochs=1000, batch_size=32, callbacks=callback_list, validation_data=(X_val,y_val))

score = model.evaluate(X_test, y_test, verbose=0)
print('the testing accuracy is',score[1])
test_image = X_test

(model.predict(test_image))
print(model.predict_classes(test_image))

for layer in model.layers:
    print(layer.name)

del model
"""
for i in range(0,73):
    labels[i]=10

for i in range(73,141):
    labels[i]=8
for i in range(141,226):
    labels[i]=5
for i in range(226,303):
    labels[i]=4
for i in range(303,373):
    labels[i]=9
for i in range(373,475):
    labels[i]=1
for i in range(475,607):
    labels[i]=7
for i in range(607,658):
    labels[i]=6
for i in range(658,783):
    labels[i]=3
for i in range(783,882):
    labels[i]=11
for i in range(882,1002):
    labels[i]=2
for i in range(1002,1120):
    labels[i]=0
"""