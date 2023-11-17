import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import numpy as np

from IPython.display import clear_output
def create_mask(pred_mask):
 pred_mask = tf.argmax(pred_mask, axis=-1)
 pred_mask = pred_mask[..., tf.newaxis]
 return pred_mask[0]
            
def resize(input_image, input_mask,shape=(128,128)):
   input_image = tf.image.resize(input_image, shape, method="nearest")
   input_mask = tf.image.resize(input_mask, shape, method="nearest")
   return input_image, input_mask

def augment(input_image, input_mask):
   if tf.random.uniform(()) > 0.5:
       # Random flipping of the image and mask
       input_image = tf.image.flip_left_right(input_image)
       input_mask = tf.image.flip_left_right(input_mask)
   return input_image, input_mask

def normalize(input_image, input_mask):
   input_image = tf.cast(input_image, tf.float32) / 255.0
   input_mask -= 1
   return input_image, input_mask

def mask_channels_expand(mask,num_classes):
   channels = []

   for c in range(num_classes): 
      channels.append(mask == c)
   
   channels = np.stack(channels,axis=-1)
   return channels
   
from tqdm import tqdm
import cv2
def convert_data(data,shape=(128,128)):
   x = []
   y = []

   for sample in tqdm(data):

      img    = sample['image']
      img    = cv2.resize(img.numpy(),shape)
      img    = img/255.0

      mask   = sample['segmentation_mask']
      mask   = cv2.resize(mask.numpy(),shape)
      #mask -= 1
      mask   = mask_channels_expand(mask,num_classes=3)

      x.append(img)
      y.append(mask)

   x = np.stack(x)
   y = np.stack(y)

   return x,y

def load_data(shape):
   train_data, info_t = tfds.load(name='oxford_iiit_pet:3.*.*', split="train",with_info=True)
   valid_data, info_v = tfds.load(name='oxford_iiit_pet:3.*.*', split="test",with_info=True)

   x_train,y_train = convert_data(train_data,shape)
   x_val,y_val     = convert_data(valid_data,shape)
   
   return x_train,x_val,y_train,y_val

def load_train(BATCH_SIZE=64):
    
    dataset, info = tfds.load('oxford_iiit_pet:3.*.*', with_info=True)
   

    train_size = len(dataset["train"])
    test_size  = len(dataset["train"])
    valid      = len(dataset["train"])


      
    train_dataset = dataset["train"].map(load_image_train, num_parallel_calls=tf.data.AUTOTUNE)
    test_dataset = dataset["test"].map(load_image_test, num_parallel_calls=tf.data.AUTOTUNE)

    BUFFER_SIZE = 1000
    train_batches = train_dataset.cache().shuffle(BUFFER_SIZE).batch(BATCH_SIZE).repeat()
    train_batches = train_batches.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    validation_batches = test_dataset.take(3000).batch(BATCH_SIZE)
    test_batches = test_dataset.skip(3000).take(669).batch(BATCH_SIZE)

    return train_batches,validation_batches,test_batches,train_size//BATCH_SIZE,3000//BATCH_SIZE,BATCH_SIZE

def display(display_list):
 plt.figure(figsize=(15, 15))

 title = ["Input Image", "True Mask", "Predicted Mask"]

 for i in range(len(display_list)):
   plt.subplot(1, len(display_list), i+1)
   plt.title(title[i])
   plt.imshow(tf.keras.utils.array_to_img(display_list[i]))
   plt.axis("off")
 plt.show()