from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
import facenet
import os
import sys
import math
import pickle
import align.detect_face
import numpy as np
import cv2
import collections


def main():


    MINSIZE = 20
    THRESHOLD = [0.6, 0.7, 0.7]
    FACTOR = 0.709
    IMAGE_SIZE = 182
    INPUT_IMAGE_SIZE = 160


    custom_classifier_path = '../Models/celeb/celeb.pkl'
    video_path = '../data/testing.mp4'
    facenet_model_path = '../Models/facenet/20180402-114759.pb'


    # Load The Custom Classifier
    with open(custom_classifier_path, 'rb') as file:
        model, class_names = pickle.load(file)
    print("Custom Classifier, Successfully loaded")


    with tf.Graph().as_default():

        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.6)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options, log_device_placement=False))

        with sess.as_default():

            # Load the model
            print('Loading feature extraction model')
            facenet.load_model(facenet_model_path)


            images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
            embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")
            phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
            embedding_size = embeddings.get_shape()[1]

            pnet, rnet, onet = align.detect_face.create_mtcnn(sess, "./align")


            people_detected = set()
            person_detected = collections.Counter()

            cap = cv2.VideoCapture(video_path)

            frame_width = int(cap.get(3))
            frame_height = int(cap.get(4))
            total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
            print("Video Resolution :  {} x {}".format(frame_width,frame_height))
            out = cv2.VideoWriter('output.avi',cv2.VideoWriter_fourcc('M','J','P','G'), 10, (frame_width,frame_height))
            print("Total Frames to Process : ",total_frames)



            while(cap.isOpened()):
                ret, frame = cap.read()
                if np.shape(frame) == ():
                    break


                bounding_boxes, _ = align.detect_face.detect_face(frame, MINSIZE, pnet, rnet, onet, THRESHOLD, FACTOR)

                faces_found = bounding_boxes.shape[0]
                try:
                    if faces_found > 0:
                        det = bounding_boxes[:, 0:4]
                        bb = np.zeros((faces_found, 4), dtype=np.int32)
                        for i in range(faces_found):
                            bb[i][0] = det[i][0]
                            bb[i][1] = det[i][1]
                            bb[i][2] = det[i][2]
                            bb[i][3] = det[i][3]

                            cropped = frame[bb[i][1]:bb[i][3], bb[i][0]:bb[i][2], :]
                            scaled = cv2.resize(cropped, (INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE), interpolation=cv2.INTER_CUBIC)
                            scaled = facenet.prewhiten(scaled)
                            scaled_reshape = scaled.reshape(-1, INPUT_IMAGE_SIZE, INPUT_IMAGE_SIZE, 3)
                            feed_dict = {images_placeholder: scaled_reshape, phase_train_placeholder: False}
                            emb_array = sess.run(embeddings, feed_dict=feed_dict)
                            predictions = model.predict_proba(emb_array)
                            best_class_indices = np.argmax(predictions, axis=1)
                            best_class_probabilities = predictions[np.arange(len(best_class_indices)), best_class_indices]
                            best_name = class_names[best_class_indices[0]]
                            print("Processing Frame : {}/{}".format(cap.get(cv2.CAP_PROP_POS_FRAMES),total_frames))
                            print("Name: {}, Probability: {} \n\n".format(best_name, best_class_probabilities))


                            cv2.rectangle(frame, (bb[i][0], bb[i][1]), (bb[i][2], bb[i][3]), (0, 255, 0), 2)
                            text_x = bb[i][0]
                            text_y = bb[i][3] + 20

                            if best_class_probabilities > 0.15:
                                name = class_names[best_class_indices[0]]
                            else:
                                name = "Unknown"
                            cv2.putText(frame, name, (text_x, text_y), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                            1, (255, 255, 255), thickness=1, lineType=2)
                            cv2.putText(frame, str(round(best_class_probabilities[0], 3)), (text_x, text_y+17), cv2.FONT_HERSHEY_COMPLEX_SMALL,
                                            1, (255, 255, 255), thickness=1, lineType=2)
                            person_detected[best_name] += 1
                except:
                    pass

                out.write(frame)

            cap.release()
            cv2.destroyAllWindows()

main()
