from pydub import AudioSegment
import os 
from pydub.silence import split_on_silence


l1=[]   
l2=[]
for(dirpath, subdir, filenames) in os.walk("C:/collegework/CDSAML/audio/"):   #review this line
       for fl in filenames:
           l1.append(fl)

l1.sort()
print(l1)
for(dirpath, subdir, filenames) in os.walk("C:/collegework/CDSAML/dataset_englishh/"):   #review this line
    for fl in subdir:
        l2.append(fl)
print(l2)
for i in range(len(l1)):
    g="prerana"+l2[i]+".wav"
    os.rename("C:/collegework/CDSAML/audio/"+l1[i],"C:/collegework/CDSAML/audio/"+g)
    b=AudioSegment.from_file("C:/collegework/CDSAML/audio/"+g,format="wav")
    b.export("C:/collegework/CDSAML/dataset_englishh/"+l2[i]+"/"+g, format="wav")