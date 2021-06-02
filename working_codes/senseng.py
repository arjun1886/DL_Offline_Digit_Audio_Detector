from pydub import AudioSegment
import os 
from pydub.silence import split_on_silence


l1=["0.wav","1.wav","2.wav","3.wav","4.wav","5.wav","6.wav","7.wav","8.wav","9.wav","10.wav","11.wav","12.wav","13.wav"]   
l2=["zero","one","two","three","four","five","six","seven","eight","nine","double","triple","hundred","thousand"]



for i in range(len(l1)):
    g="testchristy5"+l2[i]+".wav"
    os.rename("C:/collegework/CDSAML/audio/"+l1[i],"C:/collegework/CDSAML/audio/"+g)
    b=AudioSegment.from_file("C:/collegework/CDSAML/audio/"+g,format="wav")
    b.export("C:/collegework/CDSAML/dataset_englishh/"+l2[i]+"/"+g, format="wav")