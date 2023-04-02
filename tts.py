import pyttsx3

ttsengine = pyttsx3.init()


class AudioStraem():
    def __init__(self):
        super().__init__()
        self.classlabel = ''

    def playstream(self):
        ttsengine.say(self.classlabel)
        ttsengine.runAndWait()
        ttsengine.stop()
