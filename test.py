import os
import colorsys

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input

from yolo4.model import yolo_eval, yolo4_body
from yolo4.utils import letterbox_image

from PIL import Image, ImageFont, ImageDraw
from timeit import default_timer as timer
import matplotlib.pyplot as plt
import argparse
import cv2

class Yolo4(object):
    def get_class(self):
        classes_path = os.path.expanduser(self.classes_path)
        with open(classes_path) as f:
            class_names = f.readlines()
        class_names = [c.strip() for c in class_names]
        return class_names

    def get_anchors(self):
        anchors_path = os.path.expanduser(self.anchors_path)
        with open(anchors_path) as f:
            anchors = f.readline()
        anchors = [float(x) for x in anchors.split(',')]
        return np.array(anchors).reshape(-1, 2)

    def load_yolo(self):
        model_path = os.path.expanduser(self.model_path)
        assert model_path.endswith('.h5'), 'Keras model or weights must be a .h5 file.'

        self.class_names = self.get_class()
        self.anchors = self.get_anchors()
        print('self.anchors', self.anchors)

        num_anchors = len(self.anchors)
        num_classes = len(self.class_names)

        # Generate colors for drawing bounding boxes.
        hsv_tuples = [(x / len(self.class_names), 1., 1.)
                      for x in range(len(self.class_names))]
        self.colors = list(map(lambda x: colorsys.hsv_to_rgb(*x), hsv_tuples))
        self.colors = list(
            map(lambda x: (int(x[0] * 255), int(x[1] * 255), int(x[2] * 255)),
                self.colors))

        self.sess = K.get_session()

        # Load model, or construct model and load weights.
        self.yolo4_model = yolo4_body(Input(shape=(self.input_size[1], self.input_size[0], 3)), num_anchors//3, num_classes, self.alpha, True)
        self.yolo4_model.load_weights(model_path)

        print('{} model, anchors, and classes loaded.'.format(model_path))

        if self.gpu_num>=2:
            self.yolo4_model = multi_gpu_model(self.yolo4_model, gpus=self.gpu_num)

        self.input_image_shape = K.placeholder(shape=(2, ))
        self.boxes, self.scores, self.classes = yolo_eval(self.yolo4_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score)

    def __init__(self, score, iou, input_size, anchors_path, classes_path, model_path, alpha=1, gpu_num=1):
        self.score = score
        self.iou = iou
        self.input_size = input_size
        self.anchors_path = anchors_path
        self.classes_path = classes_path
        self.model_path = model_path
        self.gpu_num = gpu_num
        self.alpha = alpha
        self.load_yolo()

    def close_session(self):
        self.sess.close()

    def detect_image(self, image, model_image_size=(608, 608)):
        start = timer()

        boxed_image = letterbox_image(image, tuple(reversed(model_image_size)))
        image_data = np.array(boxed_image, dtype='float32')

        print(image_data.shape)
        image_data /= 255.
        image_data = np.expand_dims(image_data, 0)  # Add batch dimension.

        out_boxes, out_scores, out_classes = self.sess.run(
            [self.boxes, self.scores, self.classes],
            feed_dict={
                self.yolo4_model.input: image_data,
                self.input_image_shape: [image.size[1], image.size[0]],
                K.learning_phase(): 0
            })

        print('Found {} boxes for {}'.format(len(out_boxes), 'img'))

        font = ImageFont.truetype(font='FreeSerif.ttf',
                    size=np.floor(3e-2 * image.size[1] + 0.5).astype('int32'))
        thickness = (image.size[0] + image.size[1]) // 300

        for i, c in reversed(list(enumerate(out_classes))):
            predicted_class = self.class_names[c]
            box = out_boxes[i]
            score = out_scores[i]

            label = '{} {:.2f}'.format(predicted_class, score)
            draw = ImageDraw.Draw(image)
            label_size = draw.textsize(label, font)

            top, left, bottom, right = box
            top = max(0, np.floor(top + 0.5).astype('int32'))
            left = max(0, np.floor(left + 0.5).astype('int32'))
            bottom = min(image.size[1], np.floor(bottom + 0.5).astype('int32'))
            right = min(image.size[0], np.floor(right + 0.5).astype('int32'))
            print(label, (left, top), (right-left, bottom-top))

            if top - label_size[1] >= 0:
                text_origin = np.array([left, top - label_size[1]])
            else:
                text_origin = np.array([left, top + 1])

            # My kingdom for a good redistributable image drawing library.
            for i in range(thickness):
                draw.rectangle(
                    [left + i, top + i, right - i, bottom - i],
                    outline=self.colors[c])
            draw.rectangle(
                [tuple(text_origin), tuple(text_origin + label_size)],
                fill=self.colors[c])
            draw.text(text_origin, label, fill=(0, 0, 0), font=font)
            del draw

        end = timer()
        print(end - start)
        return image

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type = float, default = 1)
    parser.add_argument("--model_path", type = str, default = 'yolov4.h5')
    parser.add_argument("--anchors_path", type = str, default = 'model_data/yolo4_anchors.txt')
    parser.add_argument("--classes_path", type = str, default = 'model_data/coco_classes.txt')
    parser.add_argument("--input_size", nargs='*', type=int, default=[512, 288]) # skip images in each cats for coco
    parser.add_argument("--thd", type=float, default=0.5)
    parser.add_argument("-i", type = str, default = '')
    parser.add_argument("-o", type = str, default = '')
    args = parser.parse_args()
    print(args)

    iou = 0.5

    input_size = (args.input_size[0], args.input_size[1])

    yolo4_model = Yolo4(args.thd, iou, input_size, args.anchors_path, args.classes_path, args.model_path, alpha=args.alpha)


    if len(args.i) == 0:
        cap = cv2.VideoCapture(0)
    else:
        cap = cv2.VideoCapture(args.i)

    if (cap.isOpened() == False):
        print("Unable to read camera feed")
        exit(0)

    if len(args.o) != 0:
        videoWriter = cv2.VideoWriter(args.o, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, input_size)
        print('[video]', videoWriter)

    while True:
        ret, img = cap.read()
        if not ret:
            break

        img = cv2.resize(img, input_size, interpolation=cv2.INTER_AREA)
        img = img[...,::-1]
        result = yolo4_model.detect_image(Image.fromarray(img), model_image_size=tuple(reversed(input_size)))
        # plt.imshow(result)
        # plt.show()
        result = np.asarray(result)[...,::-1]
        cv2.imshow('r', result)
        if (cv2.waitKey(1) & 0xFF == ord('q')):
            if len(args.o) != 0:
                videoWriter.release()
            exit()

        if len(args.o) != 0:
            videoWriter.write(result)

    yolo4_model.close_session()
