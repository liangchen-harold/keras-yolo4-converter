import os
import colorsys

import numpy as np
from keras import backend as K
from keras.models import load_model
from keras.layers import Input
from keras.layers import Conv2D, Add, ZeroPadding2D, UpSampling2D, Concatenate, MaxPooling2D, Activation, Multiply
import tensorflow as tf
from tensorflow.python.framework import graph_util
K.set_learning_phase(0)

from yolo4.model import yolo_eval, yolo4_body
from yolo4.utils import letterbox_image

from PIL import Image, ImageFont, ImageDraw
from timeit import default_timer as timer
import matplotlib.pyplot as plt

from operator import itemgetter
import argparse

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

        # Read and convert darknet weight
        print('Loading weights.')
        weights_file = open(self.weights_path, 'rb')
        major, minor, revision = np.ndarray(
            shape=(3, ), dtype='int32', buffer=weights_file.read(12))
        if (major*10+minor)>=2 and major<1000 and minor<1000:
            seen = np.ndarray(shape=(1,), dtype='int64', buffer=weights_file.read(8))
        else:
            seen = np.ndarray(shape=(1,), dtype='int32', buffer=weights_file.read(4))
        print('Weights Header: ', major, minor, revision, seen)

        convs_to_load = []
        bns_to_load = []
        for i in range(len(self.yolo4_model.layers)):
            layer_name = self.yolo4_model.layers[i].name
            if layer_name.startswith('conv2d_'):
                convs_to_load.append((int(layer_name[7:]), i))
            if layer_name.startswith('batch_normalization_'):
                bns_to_load.append((int(layer_name[20:]), i))

        convs_sorted = sorted(convs_to_load, key=itemgetter(0))
        bns_sorted = sorted(bns_to_load, key=itemgetter(0))

        bn_index = 0
        for i in range(len(convs_sorted)):
            print('Converting ', i)
            if i == 93 or i == 101 or i == 109:
                #no bn, with bias
                weights_shape = self.yolo4_model.layers[convs_sorted[i][1]].get_weights()[0].shape
                bias_shape = self.yolo4_model.layers[convs_sorted[i][1]].get_weights()[0].shape[3]
                filters = bias_shape
                size = weights_shape[0]
                darknet_w_shape = (filters, weights_shape[2], size, size)
                weights_size = np.product(weights_shape)

                conv_bias = np.ndarray(
                    shape=(filters, ),
                    dtype='float32',
                    buffer=weights_file.read(filters * 4))
                conv_weights = np.ndarray(
                    shape=darknet_w_shape,
                    dtype='float32',
                    buffer=weights_file.read(weights_size * 4))
                conv_weights = np.transpose(conv_weights, [2, 3, 1, 0])
                self.yolo4_model.layers[convs_sorted[i][1]].set_weights([conv_weights, conv_bias])
            else:
                #with bn, no bias
                weights_shape = self.yolo4_model.layers[convs_sorted[i][1]].get_weights()[0].shape
                size = weights_shape[0]
                bn_shape = self.yolo4_model.layers[bns_sorted[bn_index][1]].get_weights()[0].shape
                filters = bn_shape[0]
                darknet_w_shape = (filters, weights_shape[2], size, size)
                weights_size = np.product(weights_shape)

                conv_bias = np.ndarray(
                    shape=(filters, ),
                    dtype='float32',
                    buffer=weights_file.read(filters * 4))
                bn_weights = np.ndarray(
                    shape=(3, filters),
                    dtype='float32',
                    buffer=weights_file.read(filters * 12))

                bn_weight_list = [
                    bn_weights[0],  # scale gamma
                    conv_bias,  # shift beta
                    bn_weights[1],  # running mean
                    bn_weights[2]  # running var
                ]
                self.yolo4_model.layers[bns_sorted[bn_index][1]].set_weights(bn_weight_list)

                conv_weights = np.ndarray(
                    shape=darknet_w_shape,
                    dtype='float32',
                    buffer=weights_file.read(weights_size * 4))
                conv_weights = np.transpose(conv_weights, [2, 3, 1, 0])
                self.yolo4_model.layers[convs_sorted[i][1]].set_weights([conv_weights])

                bn_index += 1

        weights_file.close()

        self.yolo4_model.save(self.model_path)

        all_output_layer = self.yolo4_model.outputs

        output_node_names = [n.name.split(':')[0] for n in all_output_layer]
        print(output_node_names)
        with open(self.output_layer_file, 'w') as f:
            f.write(' '.join(output_node_names))
        pb_path = '.'.join(self.model_path.split('.')[0:-1]) + '.pb'
        frozen_graph = self.freeze_session(output_names=output_node_names)
        tf.train.write_graph(frozen_graph, os.path.dirname(self.model_path), pb_path, as_text=False)
        writer = tf.summary.FileWriter(os.path.dirname(self.model_path), frozen_graph)
        writer.close()

        if self.gpu_num>=2:
            self.yolo4_model = multi_gpu_model(self.yolo4_model, gpus=self.gpu_num)

        self.input_image_shape = K.placeholder(shape=(2, ))
        self.boxes, self.scores, self.classes = yolo_eval(self.yolo4_model.output, self.anchors,
                len(self.class_names), self.input_image_shape,
                score_threshold=self.score)

    def freeze_session(self, output_names=None):
        graph = self.sess.graph
        with graph.as_default():
            freeze_var_names = list(set(v.op.name for v in tf.global_variables()))
            output_names = output_names or []
            output_names += [v.op.name for v in tf.global_variables()]
            # Graph -> GraphDef ProtoBuf
            input_graph_def = graph.as_graph_def()
            frozen_graph = graph_util.convert_variables_to_constants(self.sess, input_graph_def, output_names, freeze_var_names)
            return frozen_graph

    def __init__(self, score, iou, input_size, anchors_path, classes_path, model_path, weights_path, output_layer_file, alpha=1, gpu_num=1):
        self.score = score
        self.iou = iou
        self.input_size = input_size
        self.anchors_path = anchors_path
        self.classes_path = classes_path
        self.weights_path = weights_path
        self.model_path = model_path
        self.gpu_num = gpu_num
        self.alpha = alpha
        self.output_layer_file = output_layer_file
        self.load_yolo()

    def close_session(self):
        self.sess.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--alpha", type = float, default = 1)
    parser.add_argument("--model_path", type = str, default = 'yolov4.weights')
    parser.add_argument("--anchors_path", type = str, default = 'model_data/yolo4_anchors.txt')
    parser.add_argument("--classes_path", type = str, default = 'model_data/coco_classes.txt')
    parser.add_argument("--input_size", nargs='*', type=int, default=[512, 288]) # skip images in each cats for coco
    parser.add_argument("-o", type = str, default = 'yolo4_weight.h5')
    parser.add_argument("--output_layer_file", type = str, default = 'outputs.txt')
    args = parser.parse_args()
    print(args)

    score = 0.5
    iou = 0.5

    input_size = (args.input_size[0], args.input_size[1])

    yolo4_model = Yolo4(score, iou, input_size, args.anchors_path, args.classes_path, args.o, args.model_path, args.output_layer_file, alpha=args.alpha)

    yolo4_model.close_session()
