from pydub import AudioSegment
import os 
from pydub.silence import split_on_silence


l1=["0.wav","1.wav","2.wav","3.wav","4.wav","5.wav","6.wav","7.wav","8.wav","9.wav"]   
l2=["shoonya","ek","dho","theen","char","paanch","cheh","saath","aat","nau"]



for i in range(len(l1)):
    g="testchristy"+l2[i]+"4.wav"
    os.rename("C:/collegework/CDSAML/audio/"+l1[i],"C:/collegework/CDSAML/audio/"+g)
    b=AudioSegment.from_file("C:/collegework/CDSAML/audio/"+g,format="wav")
    b.export("C:/collegework/CDSAML/dataset_hindii/"+l2[i]+"/"+g, format="wav")
