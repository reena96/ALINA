#!/usr/bin/env python
"""
classify.py is an out-of-the-box image classifer callable from the command line.

By default it configures and runs the Caffe reference ImageNet model.
"""
import numpy as np
import os
import sys
import argparse
import glob
import time
import caffe
import urllib
import requests
import pandas as pd
import logging
import subprocess
import psycopg2
from config import Config
from subprocess import PIPE, STDOUT, Popen
#from subprocess import call

def main(argv):
	
    pycaffe_dir = os.path.dirname(__file__)

    parser = argparse.ArgumentParser()
    # Required arguments: input and output files.
	
	# FROM HERE Added by Reena Mary on March 5th, 2017
    parser.add_argument( 
        "--print_results", 
        action='store_true', 
        help="Write output text to stdout rather than serializing to a file." 
    ) 
    parser.add_argument( 
        "--labels_file", 
        default=os.path.join(pycaffe_dir,"../models/vgg_face_caffe/names.txt"), help="Readable label definition file." 
        #default=os.path.join(pycaffe_dir,"../models/places365/categories_places365.txt"), help="Readable label definition file."
    )
    # TO HERE
	
    parser.add_argument(
        "input_file",
        help="Input image, directory, or npy."
    )
    parser.add_argument(
        "output_file",
        help="Output npy filename."
    )
    # Optional arguments.
    parser.add_argument(
        "--model_def",
		#Commented by Reena Mary on March 5th, 2017
                #Caffenet
                default=os.path.join(pycaffe_dir,
                "../models/vgg_face_caffe/VGG_FACE_deploy.prototxt"),
				
                #Alexnet
		#        default=os.path.join(pycaffe_dir,
                # "../models/bvlc_alexnet/deploy.prototxt"),
		# FROM HERE Added by Reena Mary on March 5th, 2017
		
		#Places365
	    #    default=os.path.join(pycaffe_dir,
        #        "../models/places365/deploy_vgg16_places365.prototxt"),
		# TO HERE
        help="Model definition file."
    )
    parser.add_argument(
        "--pretrained_model",
		#Commented by Reena Mary on March 5th, 2017
                default=os.path.join(pycaffe_dir,
                 "../models/vgg_face_caffe/VGG_FACE.caffemodel"),
                #default=os.path.join(pycaffe_dir,
                #"../models/bvlc_alexnet/bvlc_alexnet.caffemodel"),
		# FROM HERE Added by Reena Mary on March 5th, 2017
		#Places365
		#default=os.path.join(pycaffe_dir,
        #        "../models/places365/vgg16_places365.caffemodel"),
		# TO HERE 
        help="Trained model weights file."
    )
    parser.add_argument(
        "--gpu",
        action='store_true',
        help="Switch for gpu computation."
    )
    parser.add_argument(
        "--center_only",
        action='store_true',
        help="Switch for prediction from center crop alone instead of " +
             "averaging predictions across crops (default)."
    )
    parser.add_argument(
        "--images_dim",
        default='256,256',
        help="Canonical 'height,width' dimensions of input images."
    )
    parser.add_argument(
        "--mean_file",
        default=os.path.join(pycaffe_dir,
                             'caffe/imagenet/ilsvrc_2012_mean.npy'),
        help="Data set image mean of [Channels x Height x Width] dimensions " +
             "(numpy array). Set to '' for no mean subtraction."
    )
    parser.add_argument(
        "--input_scale",
        type=float,
        help="Multiply input features by this scale to finish preprocessing."
    )
    parser.add_argument(
        "--raw_scale",
        type=float,
        default=255.0,
        help="Multiply raw input by this scale before preprocessing."
    )
    parser.add_argument(
        "--channel_swap",
        default='2,1,0',
        help="Order to permute input channels. The default converts " +
             "RGB -> BGR since BGR is the Caffe default by way of OpenCV."
    )
    parser.add_argument(
        "--ext",
        default='jpg',
        help="Image file extension to take as input when a directory " +
             "is given as the input file."
    )
    args = parser.parse_args()

    image_dims = [int(s) for s in args.images_dim.split(',')]

    mean, channel_swap = None, None
    if args.mean_file:
        mean = np.load(args.mean_file)
    if args.channel_swap:
        channel_swap = [int(s) for s in args.channel_swap.split(',')]

    if args.gpu: 
        caffe.set_mode_gpu()
        print("GPU mode")
    else:
        caffe.set_mode_cpu() 
        print("CPU mode")

    # Make classifier.
    classifier = caffe.Classifier(args.model_def, args.pretrained_model,
            image_dims=image_dims, mean=mean,
            input_scale=args.input_scale, raw_scale=args.raw_scale,
            channel_swap=channel_swap)

    # Load numpy array (.npy), directory glob (*.jpg), or image file.
    args.input_file = os.path.expanduser(args.input_file)
    if args.input_file.endswith('npy'):
        print("Loading file: %s" % args.input_file)
        inputs = np.load(args.input_file)
    elif os.path.isdir(args.input_file):
        print("Loading folder: %s" % args.input_file)
        inputs =[caffe.io.load_image(im_f)
                 for im_f in glob.glob(args.input_file + '/*.' + args.ext)]
    else:
        print("Loading file:------------------------- %s" % args.input_file)
        imagepathname = args.input_file
        inputs = [caffe.io.load_image(args.input_file)]

    print("Classifying %d inputs." % len(inputs))

    # Classify.
    start = time.time()
    # FROM HERE Added by Reena Mary on March 5th, 2017
    scores = classifier.predict(inputs, not args.center_only).flatten()
    # TO HERE
	
    # Commented out by Reena Mary
    #predictions = classifier.predict(inputs, not args.center_only)
    print("Done in %.2f s." % (time.time() - start))
    
    # FROM HERE Added by Reena Mary on March 5th, 2017
    if args.print_results: 
        with open(args.labels_file) as f: 
            labels_df = pd.DataFrame([{'synset_id':l.strip().split(' ')[0], 'name': ' '.join(l.strip().split(' ')[1:]).split(',')[0]} for l in f.readlines()]) 
            logging.info("---------------1 labels_df-----------")
            #labels_df = pd.DataFrame([{'name': ' '.join(l)} for l in f.readlines()])
            logging.info("---------------2 labels_df-----------")
            logging.info(labels_df)
            labels = labels_df.sort_values('synset_id')['name'].values #sort -> sort_values Changed by Reena Mary on March 5th, 2017
            #labels = labels_df.sort_values['name'].values #sort -> sort_values Changed by Reena Mary on March 5th, 2017
            
            indices =(-scores).argsort()[:1] 
            predictions = labels[indices]
            #expected_infogain = np.dot(
            #    self.bet['probmat'], scores[self.bet['idmapping']])
            #expected_infogain *= self.bet['infogain']

            # sort the scores
            #infogain_sort = expected_infogain.argsort()[::-1]
            #bet_result = [(self.bet['words'][v], '%.5f' % expected_infogain[v])
            #              for v in infogain_sort[:5]]
            #b = str(bet_result)
            #logging.info('MAXIMALLY ACCURATE BET result:---------------------- %s', b)
            meta = [(p, '%.5f' % scores[i]) for i,p in zip(indices, predictions)] 
            print(str(meta))
            #[('A.R._Rahman', '0.97070')]
            #logging.info('-------------------------------%s',p[0][0])
        with open('/home/reena-mary/caffe-face-caffe-face/python/output.txt','w') as t:
            t.write(str(meta))
            #logging.info('MAXIMALLY SPECIFIC META result:---------------------- %s', c)
           
    #TO HERE

    # Doing object detection using YOLO
        
        #req_path = "/home/reena-mary/darknet/"
        #os.chdir(req_path)
        #pathnow="/home/reena-mary/Pictures/people.jpg"
        #subprocess.Popen(["./darknet detect cfg/yolo.cfg yolo.weights "+pathnow],shell=True)
    #print("Saving results into %s" % args.output_file)
    #np.save(args.output_file, predictions)

	
if __name__ == '__main__':
    main(sys.argv)