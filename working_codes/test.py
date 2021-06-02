from pydub import AudioSegment
import os 
from pydub.silence import split_on_silence

def detect_leading_silence(sound,silence_threshold=-35.0,chunk_size=40):
    split_audio=split_on_silence(sound,min_silence_len=200,silence_thresh=-35.0,keep_silence=100
                                 )
    return split_audio

for(dirpath, subdir, filenames) in os.walk("C:/collegework/CDSAML/audio/"):   #review this line
    for fl in filenames:
        sound= AudioSegment.from_file("C:/collegework/CDSAML/audio/"+fl,format="wav")        #review this line
        #sound.apply_gain(3)
        trimmed_sound=detect_leading_silence(sound)
        
        if(len(trimmed_sound)>0):
          max=trimmed_sound[0]
          newpath="C:/collegework/CDSAML/audio/"
          if not os.path.exists(newpath):
              os.makedirs(newpath)
          for i in trimmed_sound:
              
              if i.max_dBFS>=max.max_dBFS and i.max_dBFS>-2:
                 max=i
          i.export(newpath+fl,format="wav")
    