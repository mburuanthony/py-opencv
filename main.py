import os
from detector import Detector


def main():
    video_path = 'video\Traffic-27260.mp4'
    # video_path = 0 # Use video feed from device primary camera
    config_path = os.path.join('model',
                               'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt')
    model_path = os.path.join('model', 'frozen_inference_graph.pb')
    classes_path = os.path.join('model', 'coco.names')

    detector = Detector(video_path, config_path, model_path, classes_path)
    detector.readClasses()
    detector.onVideo()


if __name__ == '__main__':
    main()
