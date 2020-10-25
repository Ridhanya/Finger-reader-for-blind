from gtts import gTTS
from playsound import playsound
import textract
txt= textract.process('sample.tif')
tts = gTTS(text=txt,lang='eng')
tts.save('sample.mp3')
playsound('sample.mp3')