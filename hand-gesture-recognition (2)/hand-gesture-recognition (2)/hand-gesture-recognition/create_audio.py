import gtts

f = open('gesture.names', 'r')
classNames = f.read().split('\n')
f.close()
print(classNames)

for classId in classNames:
    tts = gtts.gTTS(classId)
    filename = "audio/" + classId + ".mp3"
    tts.save(filename)