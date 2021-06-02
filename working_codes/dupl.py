from pydub import AudioSegment
import os 
from pydub.silence import split_on_silence



for(dirpath, subdir, filenames) in os.walk("C:/collegework/CDSAML/trimmed_eng_dup/"):  
    for i in ['a','k','l','b']:#review this line
        for fl in filenames:
            digit=dirpath.split('/')[::-1][0]
            sound= AudioSegment.from_file("C:/collegework/CDSAML/trimmed_eng_dup/"+digit+'/'+fl,format="wav")        #review this line
            fl=str(i)+fl
            newpath="C:/collegework/CDSAML/trimmed_eng_dup/"+digit+'/'
            sound.export(newpath+fl,format="wav")
    