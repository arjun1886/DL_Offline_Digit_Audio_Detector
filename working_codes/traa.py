import os
import numpy as np
import sklearn
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
import warnings
warnings.filterwarnings("ignore")


#Define data path
data_path = "C:/collegework/CDSAML/trimmed_eng_dup/"
data_dir_list = os.listdir(data_path)
print(data_dir_list)
map=[[1 for i in range(2)] for j in range(25)]

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
    elif data_dir_list[i]=='zero':
       map[i][0]='zero'
       map[i][1]=0

    elif data_dir_list[i]=='nau':
       map[i][0]='nau'
       map[i][1]=9  
    elif data_dir_list[i]=='aat':
       map[i][0]='aat'
       map[i][1]=8
    elif data_dir_list[i]=='saath':
       map[i][0]='saath'
       map[i][1]=7
    elif data_dir_list[i]=='cheh':
       map[i][0]='cheh'
       map[i][1]=6
    elif data_dir_list[i]=='paanch':
       map[i][0]='paanch'
       map[i][1]=5
    elif data_dir_list[i]=='char':
       map[i][0]='char'
       map[i][1]=4
    elif data_dir_list[i]=='theen':
       map[i][0]='theen'
       map[i][1]=3
    elif data_dir_list[i]=='dho':
       map[i][0]='dho'
       map[i][1]=2
    elif data_dir_list[i]=='ek':
       map[i][0]='ek'
       map[i][1]=1
    elif data_dir_list[i]=='shoonya':
       map[i][0]='shoonya'
       map[i][1]=0   
    else:
       map[i][0]='silence'
       map[i][1]=14
       
num_channel=1

#Define the number of classes
num_classes = 25

audio_data_list=[]
size_data=[]
lengths=[]
zeros=np.zeros((13),dtype='int64')
zeros = zeros.astype('float32')

    
max=0
for dataset in data_dir_list:
    audio_list=os.listdir(data_path+'/'+ dataset)
    "print('Loaded the images of dataset-'+'{}\n'.format(dataset))"
    lengths.append(len(audio_list))
        

    for audio in audio_list:
        X, sample_rate = librosa.load(data_path+'/'+ dataset +'/'+audio, res_type='kaiser_fast')
        #we extract mfcc feature from data
        mfccs=np.array(librosa.feature.mfcc(y=X,sr=sample_rate, n_mfcc=13).T)
        #mfccs = sklearn.preprocessing.scale(mfccs, axis=1)
        #mfccs=np.array(mfccs.T)
        mfccs=list(mfccs)
        #if len(mfccs)>max:
        #    max=len(mfccs)
        while len(mfccs)<74:
              mfccs.append(zeros)
        #mfccs=mfccs[:18]+mfccs[-17:]
        audio_data_list.append(np.array(mfccs))
#print(audio_data_list[0])
size_data.append(len(audio_list))
#print(max)
audio_data = np.array(audio_data_list)
audio_data = audio_data.astype('float32')
audio_data /= 255
#print(audio_data[0])



if num_channel==1:
	if K.image_dim_ordering()=='th':
		audio_data=np.expand_dims(audio_data, axis=1)
		print(audio_data.shape)

	else: 
		audio_data=np.expand_dims(audio_data, axis=4)
		print(audio_data.shape)

num_classes = 25

num_of_samples = audio_data.shape[0]
#print(num_of_samples)
labels = np.ones((num_of_samples,),dtype='int64')

i=1
while(i<25):
     lengths[i]=lengths[i]+lengths[i-1]
     i+=1

print(lengths)

for k in range(0,lengths[0]):
    labels[k]=map[0][1]


i=1
while(i<25):
     for j in range(lengths[i-1],lengths[i]):
         labels[j]=map[i][1]
     i+=1



Y= np_utils.to_categorical(labels,15)
x,y = shuffle(audio_data,Y, random_state=5)

#Split the dataset
X_train,X_val, y_train, y_val = train_test_split(x, y, test_size=0.30, random_state=4)
X_train,X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.10, random_state=4) 


#Defining the model
shape=X_train[0].shape
print(shape)

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
model.add(Dense(15,activation='softmax',name='op'))

adam=keras.optimizers.Adam(lr=0.0001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
model.compile(loss='categorical_crossentropy', optimizer= adam, metrics=['accuracy'])

filepath = 'C:/collegework/CDSAML/training/weights-improvement-{epoch:02d}-{val_acc:.2f}.hdf5'
checkpoint = ModelCheckpoint(filepath, monitor='val_acc',verbose=1,save_best_only=True,mode='max')
callback_list = [checkpoint]
model.fit(X_train,y_train, epochs=1, batch_size=32, callbacks=callback_list, validation_data=(X_val,y_val))

score = model.evaluate(X_test, y_test, verbose=0)
print('the testing accuracy is',score[1])
test_image = X_test

(model.predict(test_image))
print(model.predict_classes(test_image))

outputs = [layer.output for layer in model.layers]
print(outputs)

del model