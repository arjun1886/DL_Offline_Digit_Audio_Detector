from pydub import AudioSegment
import os 
from pydub.silence import split_on_silence
import math


for(dirpath, subdir, filenames) in os.walk("C:/collegework/CDSAML/trimmed_eng_dup/"):
    for fl in filenames:
        digit=dirpath.split('/')[::-1][0]
        newpath="C:/collegework/CDSAML/trimmed_eng_dup/"+digit+'/'+fl
        os.remove(newpath)
        