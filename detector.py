import numpy as np
import cv2

np.random.seed(20)


class Detector:
    """
    \nvideoPath -> path to video(file/real-time captured)
    \nconfigPath -> path to ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt
    \nmodelPath -> path to frozen_inference_graph.pb
    \nclassesPath -> path to coco.names
    """

    def __init__(self, videoPath, configPath, modelPath, classesPath):
        self.videoPath = videoPath
        self.configPath = configPath
        self.modelPath = modelPath
        self.classesPath = classesPath

        # setup detection model
        self.net_model = cv2.dnn_DetectionModel(
            self.modelPath, self.configPath)
        self.net_model.setInputSize(320, 320)
        self.net_model.setInputScale(1.0 / 127.5)
        self.net_model.setInputMean((127.5, 127.5, 127.5))
        self.net_model.setInputSwapRB(True)

    def readClasses(self):
        classes_file = open(self.classesPath, 'r')
        self.classesList = classes_file.read().splitlines()

        classes_file.close()

        self.classesList.insert(0, '__background__')
        self.colorsList = np.random.uniform(
            low=0, high=255, size=(len(self.classesList), 3))

    def onVideo(self):
        vid = cv2.VideoCapture(self.videoPath)

        if vid.isOpened() == False:
            print(f'error opening file {self.videoPath}')
            quit()

        success, image = vid.read()

        while success:
            # self.net_model.detect returns -> class labels (item), confidence levels, bounding boxes
            classLabelIDS, confidences, bboxs = self.net_model.detect(
                image, confThreshold=0.5)

            bboxs = list(bboxs)
            confidences = list(np.array(confidences).reshape(1, -1)[0])

            bboxIDx = cv2.dnn.NMSBoxes(
                bboxs, confidences, score_threshold=0.5, nms_threshold=0.2)

            if len(bboxIDx) != 0:
                for i in range(0, len(bboxIDx)):
                    bbox = bboxs[np.squeeze(bboxIDx[i])]
                    class_confidence = confidences[np.squeeze(bboxIDx[i])]
                    class_label_id = np.squeeze(
                        classLabelIDS[np.squeeze(bboxIDx[i])])
                    class_label = self.classesList[class_label_id]

                    # get random color for each bounding box
                    class_color = [int(c)
                                   for c in self.colorsList[class_label_id]]

                    # text to display
                    class_label_text = '{}:{:.4f}'.format(
                        class_label, class_confidence)

                    # get bounding box (bbox) x, y coordinates and w, h (width, height)
                    x, y, w, h = bbox
                    cv2.rectangle(image, (x, y), (x+w, y+h),
                                  color=class_color, thickness=2)
                    cv2.putText(image, class_label_text, (x, y-10),
                                cv2.FONT_HERSHEY_PLAIN, 1, class_color, 2)

                    line_width = min(int(w * 0.185), int(h * 0.185))
                    # top left corner
                    cv2.line(image, (x, y),
                             (x+line_width, y), class_color, 5)
                    cv2.line(image, (x, y),
                             (x, y+line_width), class_color, 5)
                    # top right corner
                    cv2.line(image, (x+w, y),
                             (x+w-line_width, y), class_color, 5)
                    cv2.line(image, (x+w, y),
                             (x+w, y+line_width), class_color, 5)
                    # bottom left corner
                    cv2.line(image, (x, y+h),
                             (x+line_width, y+h), class_color, 5)
                    cv2.line(image, (x, y+h),
                             (x, y+h-line_width), class_color, 5)
                    # bottom right corner
                    cv2.line(image, (x+w, y+h),
                             (x+w-line_width, y+h), class_color, 5)
                    cv2.line(image, (x+w, y+h),
                             (x+w, y+h-line_width), class_color, 5)

            cv2.imshow('object-detection', image)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break

            success, image = vid.read()
        cv2.destroyAllWindows()
