# Based on https://www.tensorflow.org/tutorials/images/transfer_learning

# Imports
import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
keras = tf.keras

import tensorflow_datasets as tfds
tfds.disable_progress_bar()

# Split the data manually into 80% training, 10% testing, 10% validation
(raw_train, raw_validation, raw_test), metadata = tfds.load(
    'cats_vs_dogs',
    split=['train[:80%]', 'train[80%:90%]', 'train[90%:]'],
    with_info=True,
    as_supervised=True,
)

get_label_name = metadata.features['label'].int2str  # creates a function object that we can use to get labels

# display 5 images from the dataset
for image, label in raw_train.take(5):
	plt.figure()
  	plt.imshow(image)
  	plt.title(get_label_name(label))


IMG_SIZE = 160 # All images will be resized to 160x160

def format_example(image, label):
	#returns an image that is reshaped to IMG_SIZE
	image = tf.cast(image, tf.float32)
	image = (image/127.5) - 1
	image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
	return image, label

# Applying the function to the whole dataset
train = raw_train.map(format_example)
validation = raw_validation.map(format_example)
test = raw_test.map(format_example)

# Checking the images. (Images do look darker, investigate)
for image, label in train.take(2):
	plt.figure()
	plt.imshow(image)
	plt.title(get_label_name(label))

# Shuffling and splitting the images in batches
BATCH_SIZE = 32
SHUFFLE_BUFFER_SIZE = 1000

train_batches = train.shuffle(SHUFFLE_BUFFER_SIZE).batch(BATCH_SIZE)
validation_batches = validation.batch(BATCH_SIZE)
test_batches = test.batch(BATCH_SIZE)

# Checking the image size
for img, label in raw_train.take(2):
	print("Original shape:", img.shape)

for img, label in train.take(2):
	print("New shape:", img.shape)


# Picking a pretrained model:
IMG_SHAPE = (IMG_SIZE, IMG_SIZE, 3)

# Create the base model from the pre-trained model MobileNet V2
base_model = tf.keras.applications.MobileNetV2(input_shape=IMG_SHAPE,
                                               include_top=False,  # We do not want the top (classification) layer!
                                               weights='imagenet')

# This is what the pretrained model looks like:
base_model.summary()
# Model: "mobilenetv2_1.00_160"
# __________________________________________________________________________________________________
# Layer (type)                    Output Shape         Param #     Connected to                     
# ==================================================================================================
# input_1 (InputLayer)            [(None, 160, 160, 3) 0                                            
# __________________________________________________________________________________________________
# Conv1 (Conv2D)                  (None, 80, 80, 32)   864         input_1[0][0]                    
# __________________________________________________________________________________________________
# bn_Conv1 (BatchNormalization)   (None, 80, 80, 32)   128         Conv1[0][0]                      
# __________________________________________________________________________________________________
# Conv1_relu (ReLU)               (None, 80, 80, 32)   0           bn_Conv1[0][0]                   
# __________________________________________________________________________________________________
# expanded_conv_depthwise (Depthw (None, 80, 80, 32)   288         Conv1_relu[0][0]                 
# __________________________________________________________________________________________________
# expanded_conv_depthwise_BN (Bat (None, 80, 80, 32)   128         expanded_conv_depthwise[0][0]    
# __________________________________________________________________________________________________
# expanded_conv_depthwise_relu (R (None, 80, 80, 32)   0           expanded_conv_depthwise_BN[0][0] 
# __________________________________________________________________________________________________
# expanded_conv_project (Conv2D)  (None, 80, 80, 16)   512         expanded_conv_depthwise_relu[0][0
# __________________________________________________________________________________________________
# expanded_conv_project_BN (Batch (None, 80, 80, 16)   64          expanded_conv_project[0][0]      
# __________________________________________________________________________________________________
# block_1_expand (Conv2D)         (None, 80, 80, 96)   1536        expanded_conv_project_BN[0][0]   
# __________________________________________________________________________________________________
# block_1_expand_BN (BatchNormali (None, 80, 80, 96)   384         block_1_expand[0][0]             
# __________________________________________________________________________________________________
# block_1_expand_relu (ReLU)      (None, 80, 80, 96)   0           block_1_expand_BN[0][0]          
# __________________________________________________________________________________________________
# block_1_pad (ZeroPadding2D)     (None, 81, 81, 96)   0           block_1_expand_relu[0][0]        
# __________________________________________________________________________________________________
# block_1_depthwise (DepthwiseCon (None, 40, 40, 96)   864         block_1_pad[0][0]                
# __________________________________________________________________________________________________
# block_1_depthwise_BN (BatchNorm (None, 40, 40, 96)   384         block_1_depthwise[0][0]          
# __________________________________________________________________________________________________
# block_1_depthwise_relu (ReLU)   (None, 40, 40, 96)   0           block_1_depthwise_BN[0][0]       
# __________________________________________________________________________________________________
# block_1_project (Conv2D)        (None, 40, 40, 24)   2304        block_1_depthwise_relu[0][0]     
# __________________________________________________________________________________________________
# block_1_project_BN (BatchNormal (None, 40, 40, 24)   96          block_1_project[0][0]            
# __________________________________________________________________________________________________
# block_2_expand (Conv2D)         (None, 40, 40, 144)  3456        block_1_project_BN[0][0]         
# __________________________________________________________________________________________________
# block_2_expand_BN (BatchNormali (None, 40, 40, 144)  576         block_2_expand[0][0]             
# __________________________________________________________________________________________________
# block_2_expand_relu (ReLU)      (None, 40, 40, 144)  0           block_2_expand_BN[0][0]          
# __________________________________________________________________________________________________
# block_2_depthwise (DepthwiseCon (None, 40, 40, 144)  1296        block_2_expand_relu[0][0]        
# __________________________________________________________________________________________________
# block_2_depthwise_BN (BatchNorm (None, 40, 40, 144)  576         block_2_depthwise[0][0]          
# __________________________________________________________________________________________________
# block_2_depthwise_relu (ReLU)   (None, 40, 40, 144)  0           block_2_depthwise_BN[0][0]       
# __________________________________________________________________________________________________
# block_2_project (Conv2D)        (None, 40, 40, 24)   3456        block_2_depthwise_relu[0][0]     
# __________________________________________________________________________________________________
# block_2_project_BN (BatchNormal (None, 40, 40, 24)   96          block_2_project[0][0]            
# __________________________________________________________________________________________________
# block_2_add (Add)               (None, 40, 40, 24)   0           block_1_project_BN[0][0]         
#                                                                  block_2_project_BN[0][0]         
# __________________________________________________________________________________________________
# block_3_expand (Conv2D)         (None, 40, 40, 144)  3456        block_2_add[0][0]                
# __________________________________________________________________________________________________
# block_3_expand_BN (BatchNormali (None, 40, 40, 144)  576         block_3_expand[0][0]             
# __________________________________________________________________________________________________
# block_3_expand_relu (ReLU)      (None, 40, 40, 144)  0           block_3_expand_BN[0][0]          
# __________________________________________________________________________________________________
# block_3_pad (ZeroPadding2D)     (None, 41, 41, 144)  0           block_3_expand_relu[0][0]        
# __________________________________________________________________________________________________
# block_3_depthwise (DepthwiseCon (None, 20, 20, 144)  1296        block_3_pad[0][0]                
# __________________________________________________________________________________________________
# block_3_depthwise_BN (BatchNorm (None, 20, 20, 144)  576         block_3_depthwise[0][0]          
# __________________________________________________________________________________________________
# block_3_depthwise_relu (ReLU)   (None, 20, 20, 144)  0           block_3_depthwise_BN[0][0]       
# __________________________________________________________________________________________________
# block_3_project (Conv2D)        (None, 20, 20, 32)   4608        block_3_depthwise_relu[0][0]     
# __________________________________________________________________________________________________
# block_3_project_BN (BatchNormal (None, 20, 20, 32)   128         block_3_project[0][0]            
# __________________________________________________________________________________________________
# block_4_expand (Conv2D)         (None, 20, 20, 192)  6144        block_3_project_BN[0][0]         
# __________________________________________________________________________________________________
# block_4_expand_BN (BatchNormali (None, 20, 20, 192)  768         block_4_expand[0][0]             
# __________________________________________________________________________________________________
# block_4_expand_relu (ReLU)      (None, 20, 20, 192)  0           block_4_expand_BN[0][0]          
# __________________________________________________________________________________________________
# block_4_depthwise (DepthwiseCon (None, 20, 20, 192)  1728        block_4_expand_relu[0][0]        
# __________________________________________________________________________________________________
# block_4_depthwise_BN (BatchNorm (None, 20, 20, 192)  768         block_4_depthwise[0][0]          
# __________________________________________________________________________________________________
# block_4_depthwise_relu (ReLU)   (None, 20, 20, 192)  0           block_4_depthwise_BN[0][0]       
# __________________________________________________________________________________________________
# block_4_project (Conv2D)        (None, 20, 20, 32)   6144        block_4_depthwise_relu[0][0]     
# __________________________________________________________________________________________________
# block_4_project_BN (BatchNormal (None, 20, 20, 32)   128         block_4_project[0][0]            
# __________________________________________________________________________________________________
# block_4_add (Add)               (None, 20, 20, 32)   0           block_3_project_BN[0][0]         
#                                                                  block_4_project_BN[0][0]         
# __________________________________________________________________________________________________
# block_5_expand (Conv2D)         (None, 20, 20, 192)  6144        block_4_add[0][0]                
# __________________________________________________________________________________________________
# block_5_expand_BN (BatchNormali (None, 20, 20, 192)  768         block_5_expand[0][0]             
# __________________________________________________________________________________________________
# block_5_expand_relu (ReLU)      (None, 20, 20, 192)  0           block_5_expand_BN[0][0]          
# __________________________________________________________________________________________________
# block_5_depthwise (DepthwiseCon (None, 20, 20, 192)  1728        block_5_expand_relu[0][0]        
# __________________________________________________________________________________________________
# block_5_depthwise_BN (BatchNorm (None, 20, 20, 192)  768         block_5_depthwise[0][0]          
# __________________________________________________________________________________________________
# block_5_depthwise_relu (ReLU)   (None, 20, 20, 192)  0           block_5_depthwise_BN[0][0]       
# __________________________________________________________________________________________________
# block_5_project (Conv2D)        (None, 20, 20, 32)   6144        block_5_depthwise_relu[0][0]     
# __________________________________________________________________________________________________
# block_5_project_BN (BatchNormal (None, 20, 20, 32)   128         block_5_project[0][0]            
# __________________________________________________________________________________________________
# block_5_add (Add)               (None, 20, 20, 32)   0           block_4_add[0][0]                
#                                                                  block_5_project_BN[0][0]         
# __________________________________________________________________________________________________
# block_6_expand (Conv2D)         (None, 20, 20, 192)  6144        block_5_add[0][0]                
# __________________________________________________________________________________________________
# block_6_expand_BN (BatchNormali (None, 20, 20, 192)  768         block_6_expand[0][0]             
# __________________________________________________________________________________________________
# block_6_expand_relu (ReLU)      (None, 20, 20, 192)  0           block_6_expand_BN[0][0]          
# __________________________________________________________________________________________________
# block_6_pad (ZeroPadding2D)     (None, 21, 21, 192)  0           block_6_expand_relu[0][0]        
# __________________________________________________________________________________________________
# block_6_depthwise (DepthwiseCon (None, 10, 10, 192)  1728        block_6_pad[0][0]                
# __________________________________________________________________________________________________
# block_6_depthwise_BN (BatchNorm (None, 10, 10, 192)  768         block_6_depthwise[0][0]          
# __________________________________________________________________________________________________
# block_6_depthwise_relu (ReLU)   (None, 10, 10, 192)  0           block_6_depthwise_BN[0][0]       
# __________________________________________________________________________________________________
# block_6_project (Conv2D)        (None, 10, 10, 64)   12288       block_6_depthwise_relu[0][0]     
# __________________________________________________________________________________________________
# block_6_project_BN (BatchNormal (None, 10, 10, 64)   256         block_6_project[0][0]            
# __________________________________________________________________________________________________
# block_7_expand (Conv2D)         (None, 10, 10, 384)  24576       block_6_project_BN[0][0]         
# __________________________________________________________________________________________________
# block_7_expand_BN (BatchNormali (None, 10, 10, 384)  1536        block_7_expand[0][0]             
# __________________________________________________________________________________________________
# block_7_expand_relu (ReLU)      (None, 10, 10, 384)  0           block_7_expand_BN[0][0]          
# __________________________________________________________________________________________________
# block_7_depthwise (DepthwiseCon (None, 10, 10, 384)  3456        block_7_expand_relu[0][0]        
# __________________________________________________________________________________________________
# block_7_depthwise_BN (BatchNorm (None, 10, 10, 384)  1536        block_7_depthwise[0][0]          
# __________________________________________________________________________________________________
# block_7_depthwise_relu (ReLU)   (None, 10, 10, 384)  0           block_7_depthwise_BN[0][0]       
# __________________________________________________________________________________________________
# block_7_project (Conv2D)        (None, 10, 10, 64)   24576       block_7_depthwise_relu[0][0]     
# __________________________________________________________________________________________________
# block_7_project_BN (BatchNormal (None, 10, 10, 64)   256         block_7_project[0][0]            
# __________________________________________________________________________________________________
# block_7_add (Add)               (None, 10, 10, 64)   0           block_6_project_BN[0][0]         
#                                                                  block_7_project_BN[0][0]         
# __________________________________________________________________________________________________
# block_8_expand (Conv2D)         (None, 10, 10, 384)  24576       block_7_add[0][0]                
# __________________________________________________________________________________________________
# block_8_expand_BN (BatchNormali (None, 10, 10, 384)  1536        block_8_expand[0][0]             
# __________________________________________________________________________________________________
# block_8_expand_relu (ReLU)      (None, 10, 10, 384)  0           block_8_expand_BN[0][0]          
# __________________________________________________________________________________________________
# block_8_depthwise (DepthwiseCon (None, 10, 10, 384)  3456        block_8_expand_relu[0][0]        
# __________________________________________________________________________________________________
# block_8_depthwise_BN (BatchNorm (None, 10, 10, 384)  1536        block_8_depthwise[0][0]          
# __________________________________________________________________________________________________
# block_8_depthwise_relu (ReLU)   (None, 10, 10, 384)  0           block_8_depthwise_BN[0][0]       
# __________________________________________________________________________________________________
# block_8_project (Conv2D)        (None, 10, 10, 64)   24576       block_8_depthwise_relu[0][0]     
# __________________________________________________________________________________________________
# block_8_project_BN (BatchNormal (None, 10, 10, 64)   256         block_8_project[0][0]            
# __________________________________________________________________________________________________
# block_8_add (Add)               (None, 10, 10, 64)   0           block_7_add[0][0]                
#                                                                  block_8_project_BN[0][0]         
# __________________________________________________________________________________________________
# block_9_expand (Conv2D)         (None, 10, 10, 384)  24576       block_8_add[0][0]                
# __________________________________________________________________________________________________
# block_9_expand_BN (BatchNormali (None, 10, 10, 384)  1536        block_9_expand[0][0]             
# __________________________________________________________________________________________________
# block_9_expand_relu (ReLU)      (None, 10, 10, 384)  0           block_9_expand_BN[0][0]          
# __________________________________________________________________________________________________
# block_9_depthwise (DepthwiseCon (None, 10, 10, 384)  3456        block_9_expand_relu[0][0]        
# __________________________________________________________________________________________________
# block_9_depthwise_BN (BatchNorm (None, 10, 10, 384)  1536        block_9_depthwise[0][0]          
# __________________________________________________________________________________________________
# block_9_depthwise_relu (ReLU)   (None, 10, 10, 384)  0           block_9_depthwise_BN[0][0]       
# __________________________________________________________________________________________________
# block_9_project (Conv2D)        (None, 10, 10, 64)   24576       block_9_depthwise_relu[0][0]     
# __________________________________________________________________________________________________
# block_9_project_BN (BatchNormal (None, 10, 10, 64)   256         block_9_project[0][0]            
# __________________________________________________________________________________________________
# block_9_add (Add)               (None, 10, 10, 64)   0           block_8_add[0][0]                
#                                                                  block_9_project_BN[0][0]         
# __________________________________________________________________________________________________
# block_10_expand (Conv2D)        (None, 10, 10, 384)  24576       block_9_add[0][0]                
# __________________________________________________________________________________________________
# block_10_expand_BN (BatchNormal (None, 10, 10, 384)  1536        block_10_expand[0][0]            
# __________________________________________________________________________________________________
# block_10_expand_relu (ReLU)     (None, 10, 10, 384)  0           block_10_expand_BN[0][0]         
# __________________________________________________________________________________________________
# block_10_depthwise (DepthwiseCo (None, 10, 10, 384)  3456        block_10_expand_relu[0][0]       
# __________________________________________________________________________________________________
# block_10_depthwise_BN (BatchNor (None, 10, 10, 384)  1536        block_10_depthwise[0][0]         
# __________________________________________________________________________________________________
# block_10_depthwise_relu (ReLU)  (None, 10, 10, 384)  0           block_10_depthwise_BN[0][0]      
# __________________________________________________________________________________________________
# block_10_project (Conv2D)       (None, 10, 10, 96)   36864       block_10_depthwise_relu[0][0]    
# __________________________________________________________________________________________________
# block_10_project_BN (BatchNorma (None, 10, 10, 96)   384         block_10_project[0][0]           
# __________________________________________________________________________________________________
# block_11_expand (Conv2D)        (None, 10, 10, 576)  55296       block_10_project_BN[0][0]        
# __________________________________________________________________________________________________
# block_11_expand_BN (BatchNormal (None, 10, 10, 576)  2304        block_11_expand[0][0]            
# __________________________________________________________________________________________________
# block_11_expand_relu (ReLU)     (None, 10, 10, 576)  0           block_11_expand_BN[0][0]         
# __________________________________________________________________________________________________
# block_11_depthwise (DepthwiseCo (None, 10, 10, 576)  5184        block_11_expand_relu[0][0]       
# __________________________________________________________________________________________________
# block_11_depthwise_BN (BatchNor (None, 10, 10, 576)  2304        block_11_depthwise[0][0]         
# __________________________________________________________________________________________________
# block_11_depthwise_relu (ReLU)  (None, 10, 10, 576)  0           block_11_depthwise_BN[0][0]      
# __________________________________________________________________________________________________
# block_11_project (Conv2D)       (None, 10, 10, 96)   55296       block_11_depthwise_relu[0][0]    
# __________________________________________________________________________________________________
# block_11_project_BN (BatchNorma (None, 10, 10, 96)   384         block_11_project[0][0]           
# __________________________________________________________________________________________________
# block_11_add (Add)              (None, 10, 10, 96)   0           block_10_project_BN[0][0]        
#                                                                  block_11_project_BN[0][0]        
# __________________________________________________________________________________________________
# block_12_expand (Conv2D)        (None, 10, 10, 576)  55296       block_11_add[0][0]               
# __________________________________________________________________________________________________
# block_12_expand_BN (BatchNormal (None, 10, 10, 576)  2304        block_12_expand[0][0]            
# __________________________________________________________________________________________________
# block_12_expand_relu (ReLU)     (None, 10, 10, 576)  0           block_12_expand_BN[0][0]         
# __________________________________________________________________________________________________
# block_12_depthwise (DepthwiseCo (None, 10, 10, 576)  5184        block_12_expand_relu[0][0]       
# __________________________________________________________________________________________________
# block_12_depthwise_BN (BatchNor (None, 10, 10, 576)  2304        block_12_depthwise[0][0]         
# __________________________________________________________________________________________________
# block_12_depthwise_relu (ReLU)  (None, 10, 10, 576)  0           block_12_depthwise_BN[0][0]      
# __________________________________________________________________________________________________
# block_12_project (Conv2D)       (None, 10, 10, 96)   55296       block_12_depthwise_relu[0][0]    
# __________________________________________________________________________________________________
# block_12_project_BN (BatchNorma (None, 10, 10, 96)   384         block_12_project[0][0]           
# __________________________________________________________________________________________________
# block_12_add (Add)              (None, 10, 10, 96)   0           block_11_add[0][0]               
#                                                                  block_12_project_BN[0][0]        
# __________________________________________________________________________________________________
# block_13_expand (Conv2D)        (None, 10, 10, 576)  55296       block_12_add[0][0]               
# __________________________________________________________________________________________________
# block_13_expand_BN (BatchNormal (None, 10, 10, 576)  2304        block_13_expand[0][0]            
# __________________________________________________________________________________________________
# block_13_expand_relu (ReLU)     (None, 10, 10, 576)  0           block_13_expand_BN[0][0]         
# __________________________________________________________________________________________________
# block_13_pad (ZeroPadding2D)    (None, 11, 11, 576)  0           block_13_expand_relu[0][0]       
# __________________________________________________________________________________________________
# block_13_depthwise (DepthwiseCo (None, 5, 5, 576)    5184        block_13_pad[0][0]               
# __________________________________________________________________________________________________
# block_13_depthwise_BN (BatchNor (None, 5, 5, 576)    2304        block_13_depthwise[0][0]         
# __________________________________________________________________________________________________
# block_13_depthwise_relu (ReLU)  (None, 5, 5, 576)    0           block_13_depthwise_BN[0][0]      
# __________________________________________________________________________________________________
# block_13_project (Conv2D)       (None, 5, 5, 160)    92160       block_13_depthwise_relu[0][0]    
# __________________________________________________________________________________________________
# block_13_project_BN (BatchNorma (None, 5, 5, 160)    640         block_13_project[0][0]           
# __________________________________________________________________________________________________
# block_14_expand (Conv2D)        (None, 5, 5, 960)    153600      block_13_project_BN[0][0]        
# __________________________________________________________________________________________________
# block_14_expand_BN (BatchNormal (None, 5, 5, 960)    3840        block_14_expand[0][0]            
# __________________________________________________________________________________________________
# block_14_expand_relu (ReLU)     (None, 5, 5, 960)    0           block_14_expand_BN[0][0]         
# __________________________________________________________________________________________________
# block_14_depthwise (DepthwiseCo (None, 5, 5, 960)    8640        block_14_expand_relu[0][0]       
# __________________________________________________________________________________________________
# block_14_depthwise_BN (BatchNor (None, 5, 5, 960)    3840        block_14_depthwise[0][0]         
# __________________________________________________________________________________________________
# block_14_depthwise_relu (ReLU)  (None, 5, 5, 960)    0           block_14_depthwise_BN[0][0]      
# __________________________________________________________________________________________________
# block_14_project (Conv2D)       (None, 5, 5, 160)    153600      block_14_depthwise_relu[0][0]    
# __________________________________________________________________________________________________
# block_14_project_BN (BatchNorma (None, 5, 5, 160)    640         block_14_project[0][0]           
# __________________________________________________________________________________________________
# block_14_add (Add)              (None, 5, 5, 160)    0           block_13_project_BN[0][0]        
#                                                                  block_14_project_BN[0][0]        
# __________________________________________________________________________________________________
# block_15_expand (Conv2D)        (None, 5, 5, 960)    153600      block_14_add[0][0]               
# __________________________________________________________________________________________________
# block_15_expand_BN (BatchNormal (None, 5, 5, 960)    3840        block_15_expand[0][0]            
# __________________________________________________________________________________________________
# block_15_expand_relu (ReLU)     (None, 5, 5, 960)    0           block_15_expand_BN[0][0]         
# __________________________________________________________________________________________________
# block_15_depthwise (DepthwiseCo (None, 5, 5, 960)    8640        block_15_expand_relu[0][0]       
# __________________________________________________________________________________________________
# block_15_depthwise_BN (BatchNor (None, 5, 5, 960)    3840        block_15_depthwise[0][0]         
# __________________________________________________________________________________________________
# block_15_depthwise_relu (ReLU)  (None, 5, 5, 960)    0           block_15_depthwise_BN[0][0]      
# __________________________________________________________________________________________________
# block_15_project (Conv2D)       (None, 5, 5, 160)    153600      block_15_depthwise_relu[0][0]    
# __________________________________________________________________________________________________
# block_15_project_BN (BatchNorma (None, 5, 5, 160)    640         block_15_project[0][0]           
# __________________________________________________________________________________________________
# block_15_add (Add)              (None, 5, 5, 160)    0           block_14_add[0][0]               
#                                                                  block_15_project_BN[0][0]        
# __________________________________________________________________________________________________
# block_16_expand (Conv2D)        (None, 5, 5, 960)    153600      block_15_add[0][0]               
# __________________________________________________________________________________________________
# block_16_expand_BN (BatchNormal (None, 5, 5, 960)    3840        block_16_expand[0][0]            
# __________________________________________________________________________________________________
# block_16_expand_relu (ReLU)     (None, 5, 5, 960)    0           block_16_expand_BN[0][0]         
# __________________________________________________________________________________________________
# block_16_depthwise (DepthwiseCo (None, 5, 5, 960)    8640        block_16_expand_relu[0][0]       
# __________________________________________________________________________________________________
# block_16_depthwise_BN (BatchNor (None, 5, 5, 960)    3840        block_16_depthwise[0][0]         
# __________________________________________________________________________________________________
# block_16_depthwise_relu (ReLU)  (None, 5, 5, 960)    0           block_16_depthwise_BN[0][0]      
# __________________________________________________________________________________________________
# block_16_project (Conv2D)       (None, 5, 5, 320)    307200      block_16_depthwise_relu[0][0]    
# __________________________________________________________________________________________________
# block_16_project_BN (BatchNorma (None, 5, 5, 320)    1280        block_16_project[0][0]           
# __________________________________________________________________________________________________
# Conv_1 (Conv2D)                 (None, 5, 5, 1280)   409600      block_16_project_BN[0][0]        
# __________________________________________________________________________________________________
# Conv_1_bn (BatchNormalization)  (None, 5, 5, 1280)   5120        Conv_1[0][0]                     
# __________________________________________________________________________________________________
# out_relu (ReLU)                 (None, 5, 5, 1280)   0           Conv_1_bn[0][0]                  
# ==================================================================================================
# Total params: 2,257,984
# Trainable params: 2,223,872
# Non-trainable params: 34,112
# __________________________________________________________________________________________________

# We can see there are over 2.2M trainable params at the moment! We need to reduce that soon!

# At this point this base_model will simply output a shape (32, 5, 5, 1280) tensor that is a feature extraction
# from our original (1, 160, 160, 3) image. The 32 means that we have 32 layers of different filters/features.
for image, _ in train_batches.take(1):
   pass

feature_batch = base_model(image)
print(feature_batch.shape)

# We will make the model untrainable, as it is already trained
base_model.trainable = False
# If we use base_model.summary() again, we will see the trainable params is 0


# We now create our layers:
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
prediction_layer = keras.layers.Dense(1)

# And combine everything together into a model with Keras:
model = tf.keras.Sequential([
  base_model,
  global_average_layer,
  prediction_layer
])

# This is our full model now:
model.summary()
# Model: "sequential_2"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# mobilenetv2_1.00_160 (Functi (None, 5, 5, 1280)        2257984   
# _________________________________________________________________
# global_average_pooling2d (Gl (None, 1280)              0         
# _________________________________________________________________
# dense_4 (Dense)              (None, 1)                 1281      
# =================================================================
# Total params: 2,259,265
# Trainable params: 1,281
# Non-trainable params: 2,257,984
# _________________________________________________________________

# The trainable params are only now the layers that we added on top of the base model, much less than before


# Training the model:
base_learning_rate = 0.0001  # Very small learning rate to ensure that there are not really major changes to the model
model.compile(optimizer=tf.keras.optimizers.RMSprop(lr=base_learning_rate),
              loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
              metrics=['accuracy'])

# We can evaluate the model right now to see how it does before training it on our new images
initial_epochs = 3
validation_steps=20

loss0,accuracy0 = model.evaluate(validation_batches, steps = validation_steps)

# Now we can train it on our images
history = model.fit(train_batches,
                    epochs=initial_epochs,
                    validation_data=validation_batches)

# This accuracy is much higher now, almost 98%!
acc = history.history['accuracy']
print(acc)


# We can also save and load models using Keras:
model.save("dogs_vs_cats.h5")  # we can save the model and reload it at anytime in the future
new_model = tf.keras.models.load_model("dogs_vs_cats.h5")
