
import random
from tqdm import tqdm
import sys
import numpy as np
import shutil
import cv2
import pandas as pd
import os
import tensorflow as tf
import io
from PIL import Image, ImageDraw, ImageFont


def int64_feature(value):
  return tf.compat.v1.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def int64_list_feature(value):
  return tf.compat.v1.train.Feature(int64_list=tf.train.Int64List(value=value))


def bytes_feature(value):
  return tf.compat.v1.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def bytes_list_feature(value):
  return tf.compat.v1.train.Feature(bytes_list=tf.train.BytesList(value=value))


def float_feature(value):
  return tf.compat.v1.train.Feature(float_list=tf.train.FloatList(value=[value]))


def float_list_feature(value):
  return tf.compat.v1.train.Feature(float_list=tf.train.FloatList(value=value))




input_shape = (416,416) # multiple of 32, hw

def convert_dateset(root,output_file):
  #LOAD ANNOTATIONS
  annotation_path = f'{root}/_annotations.txt'
  with open(annotation_path) as f:
      lines = f.readlines()
      random.shuffle(lines)

  print(f"Found {len(lines)} elements in {annotation_path}")


  csv_lines = []

  classi = ["biker","car","pedestrian","trafficLight","truck"]

  def create_tf_example(line, path):

      line = annotated_line.split()
      img_name = line[0]
      boxes = np.array([np.array(list(map(int,box.split(',')))) for box in line[1:]])

      #print(os.path.join(path, '{}'.format(img_name)))
      with tf.compat.v1.gfile.GFile(os.path.join(path, '{}'.format(img_name)), 'rb') as fid:
          encoded_jpg = fid.read()
      encoded_jpg_io = io.BytesIO(encoded_jpg)
      image = Image.open(encoded_jpg_io)
      width, height = image.size

      filename = img_name.encode('utf8')
      image_format = 'jpg'.encode()
      xmins = []
      xmaxs = []
      ymins = []
      ymaxs = []
      classes_text = []
      classes = []

      for box in boxes:
          xmin,ymin,xmax,ymax = box[:4]
          class_idx = box[4]
          xmins.append(xmin / width)
          xmaxs.append(xmax / width)
          ymins.append(ymin / height)
          ymaxs.append(ymax / height)

          classes_text.append(classi[class_idx].encode('utf8'))
          classes.append(int(box[class_idx])+1) #CHECK IF NEED +1

      tf_example = tf.compat.v1.train.Example(features=tf.train.Features(feature={
          'image/height': int64_feature(height),
          'image/width': int64_feature(width),
          'image/filename': bytes_feature(filename),
          'image/source_id': bytes_feature(filename),
          'image/encoded': bytes_feature(encoded_jpg),
          'image/format': bytes_feature(image_format),
          'image/object/bbox/xmin': float_list_feature(xmins),
          'image/object/bbox/xmax': float_list_feature(xmaxs),
          'image/object/bbox/ymin': float_list_feature(ymins),
          'image/object/bbox/ymax': float_list_feature(ymaxs),
          'image/object/class/text': bytes_list_feature(classes_text),
          'image/object/class/label': int64_list_feature(classes),
      }))
      return tf_example

  writer = tf.compat.v1.python_io.TFRecordWriter(output_file)

  for annotated_line in tqdm(lines):
      tf_example = create_tf_example(annotated_line, root)
      writer.write(tf_example.SerializeToString())
  writer.close()

convert_dateset('./Self-Driving-Car-3/train',f'./Self-Driving-Car-3/train.tfrecord')
convert_dateset('./Self-Driving-Car-3/valid',f'./Self-Driving-Car-3/test.tfrecord')

