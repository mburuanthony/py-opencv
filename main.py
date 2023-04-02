from detector import Detector
from tts import AudioStraem
import threading
import os


def main():
    video_path = 0
    config_path = os.path.join('model',
                               'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt')
    model_path = os.path.join('model', 'frozen_inference_graph.pb')
    classes_path = os.path.join('model', 'coco.names')

    videostream = Detector(video_path, config_path, model_path, classes_path)
    audiostream = AudioStraem()

    video_thread = threading.Thread(target=videostream.onVideo)
    audio_thread = threading.Thread(target=audiostream.playstream)

    videostream.readClasses()
    audiostream.classlabel = videostream.objectclasslabel

    video_thread.start()
    audio_thread.start()

    video_thread.join()
    audio_thread.join()


if __name__ == '__main__':
    main()
