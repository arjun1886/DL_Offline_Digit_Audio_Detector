from pydub import AudioSegment
import os

for(dirpath, subdir, filenames) in os.walk("C:/collegework/CDSAML2/dataset_malayalam_alan/"):   #review this line
    for fl in filenames:
        digit=dirpath.split('/')[::-1][0]
        sound= AudioSegment.from_file("C:/collegework/CDSAML2/dataset_malayalam_alan/"+digit+'/'+fl,format="wav")        #review this line
        f="C:/collegework/CDSAML2/dataset_malayalam_alan/"+digit+'/'+digit+"alan"+str(i)+".wav"
        sound.export(f, format="wav")
            
        