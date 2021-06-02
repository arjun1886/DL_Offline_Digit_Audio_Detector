from pydub import AudioSegment
import os 
from pydub.silence import split_on_silence
import math


for(dirpath, subdir, filenames) in os.walk("C:/collegework/CDSAML2/trimmed_all_dupl/"):
    for fl in filenames:
        digit=dirpath.split('/')[::-1][0]
        sound= AudioSegment.from_file("C:/collegework/CDSAML2/trimmed_all_dupl/"+digit+'/'+fl,format="wav")
        f1=fl.split(".")[0]
        if len(f1)>=20:
           f1=f1[-(len(f1)-5):]
        elif len(f1)>=15:
             f1=f1[-(len(f1)-4):]
        elif len(f1)>=10:
             f1=f1[-(len(f1)-3):]
        else:
             f1=f1[-(len(f1)-2):]
        newpath="C:/collegework/CDSAML2/trimmed_nondup/"+digit+'/'+f1+".wav"
        sound.export(newpath,format="wav")
        