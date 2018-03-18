#!/usr/bin/env python2
#
# Example to classify faces.
# Brandon Amos
# 2015/10/11
#
# Copyright 2015-2016 Carnegie Mellon University
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import time

start = time.time()

import argparse
import cv2
import os
import pickle

from operator import itemgetter

import numpy as np
np.set_printoptions(precision=2)
import pandas as pd

import openface

from sklearn.pipeline import Pipeline
from sklearn.lda import LDA
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.mixture import GMM

#fileDir = os.path.dirname(os.path.realpath(__file__))
fileDir = '/home/lethic/projects/openface/'
modelDir = os.path.join(fileDir, 'models')
dlibModelDir = os.path.join(modelDir, 'dlib')
openfaceModelDir = os.path.join(modelDir, 'openface')

verbose = True

faceCascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
video_capture = cv2.VideoCapture(0)

def getRep(img):
    start = time.time()
    #bgrImg = cv2.imread(imgPath)
    classifier_model = '/home/lethic/projects/openface/classify-jacob-lingkang/features/classifier.pkl'
    with open(classifier_model, 'r') as f:
        (le, clf) = pickle.load(f)

    bgrImg = img
    if bgrImg is None:
        raise Exception("Unable to load image: {}".format(imgPath))

    rgbImg = cv2.cvtColor(bgrImg, cv2.COLOR_BGR2RGB)

    if verbose:
        print("  + Original size: {}".format(rgbImg.shape))
    if verbose:
        print("Loading the image took {} seconds.".format(time.time() - start))

    start = time.time()

    name = 'notnone'
    bb = align.getLargestFaceBoundingBox(rgbImg)
    if bb is None:
        name = 'none'
        #raise Exception("Unable to find a face: {}".format(imgPath))
    if verbose:
        print("Face detection took {} seconds.".format(time.time() - start))

    if name != 'none':
        start = time.time()
        alignedFace = align.align(args.imgDim, rgbImg, bb,
                                  landmarkIndices=openface.AlignDlib.OUTER_EYES_AND_NOSE)
        if alignedFace is None:
            raise Exception("Unable to align image: {}".format(imgPath))
        if verbose:
            print("Alignment took {} seconds.".format(time.time() - start))

        start = time.time()
        rep = net.forward(alignedFace)
        if verbose:
            print("Neural network forward pass took {} seconds.".format(time.time() - start))
        rep = rep.reshape(1, -1)
        start = time.time()
        predictions = clf.predict_proba(rep).ravel()
        maxI = np.argmax(predictions)
        person = le.inverse_transform(maxI)
        confidence = predictions[maxI]
        if verbose:
            print("Prediction took {} seconds.".format(time.time() - start))
        print("Predict {} with {:.2f} confidence.".format(person, confidence))
        if isinstance(clf, GMM):
            dist = np.linalg.norm(rep - clf.means_[maxI])
            print("  + Distance from the mean: {}".format(dist))
        return person
    return 'none'

def infer(args, img):
    classifier_model = '/home/lethic/projects/openface/classify-jacob-lingkang/features/classifier.pkl'
    with open(classifier_model, 'r') as f:
        (le, clf) = pickle.load(f)

    #img = '/home/lethic/projects/openface/classify-jacob-lingkang/test/Karl-small.png'
    print("\n=== {} ===".format(img))
    rep = 'none'
    if getRep(img, rep):
        rep = rep.reshape(1, -1)
        start = time.time()
        predictions = clf.predict_proba(rep).ravel()
        maxI = np.argmax(predictions)
        person = le.inverse_transform(maxI)
        confidence = predictions[maxI]
        if verbose:
            print("Prediction took {} seconds.".format(time.time() - start))
        print("Predict {} with {:.2f} confidence.".format(person, confidence))
        if isinstance(clf, GMM):
            dist = np.linalg.norm(rep - clf.means_[maxI])
            print("  + Distance from the mean: {}".format(dist))
        return person

    return 'none'

if __name__ == '__main__':

    parser = argparse.ArgumentParser()

    parser.add_argument('--dlibFacePredictor', type=str,
                        help="Path to dlib's face predictor.",
                        default=os.path.join(dlibModelDir,
                                             "shape_predictor_68_face_landmarks.dat"))
    parser.add_argument('--networkModel', type=str,
                        help="Path to Torch network model.",
                        default=os.path.join(openfaceModelDir, 'nn4.small2.v1.t7'))
    parser.add_argument('--imgDim', type=int,
                        help="Default image dimension.", default=96)
    parser.add_argument('--cuda', action='store_true')

    args = parser.parse_args()
    if verbose:
        print("Argument parsing and import libraries took {} seconds.".format(time.time() - start))

    start = time.time()

    align = openface.AlignDlib(args.dlibFacePredictor)
    net = openface.TorchNeuralNet(args.networkModel, imgDim=args.imgDim,
                                  cuda=args.cuda)

    if verbose:
        print("Loading the dlib and OpenFace models took {} seconds.".format(time.time() - start))
        start = time.time()

    while True:
        ret, frame = video_capture.read()

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.cv.CV_HAAR_SCALE_IMAGE
        )

        person = 'none'
        # Draw a rectangle around the faces
        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
            faceimg = frame[y:y+h, x:x+w]
            if w > 100.0:
                scale = 100.0/w
                faceimg = cv2.resize(faceimg, (0,0), fx=scale, fy=scale)
            person = getRep(faceimg)
            cv2.imshow('Video1', faceimg)
            cv2.putText(frame, person, (x+w, y+h), cv2.FONT_HERSHEY_SIMPLEX, 1, (255,255,255))

        # Display the resulting frame
        cv2.imshow('Video', frame)


        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # When everything is done, release the capture
    video_capture.release()
    cv2.destroyAllWindows()
