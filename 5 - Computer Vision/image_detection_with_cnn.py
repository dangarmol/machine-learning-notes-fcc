import tensorflow as tf

from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt


# Load and split dataset. This has a slightly different structure than usual as it uses a dataset instead of a np array
(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()

# Normalize pixel values to be between 0 and 1
train_images, test_images = train_images / 255.0, test_images / 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

# Let's look at one image
IMG_INDEX = 1  # Change this to look at other images

plt.imshow(train_images[IMG_INDEX], cmap=plt.cm.binary)
plt.xlabel(class_names[train_labels[IMG_INDEX][0]])
plt.show()


# CNN Architecture
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))  # 32 filters of 3x3, and input shape of 32px*32px*3channels
model.add(layers.MaxPooling2D((2, 2)))  # Pooling layer of 2x2 squares with 2 strides. Pooling layers are not strictly required!
model.add(layers.Conv2D(64, (3, 3), activation='relu'))  # Same as first one but 64 filters
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))

model.add(layers.Flatten())  # We flatten the output of the CNN
model.add(layers.Dense(64, activation='relu'))  # Add a first hidden layer
model.add(layers.Dense(10))  # And finally the output layer (one neuron per class)


# This is what our model looks like:
model.summary()

# Model: "sequential_2"
# ______________________________________________________________
# Layer (type)                 Output Shape              Param #   
# ==============================================================
# conv2d_6 (Conv2D)            (None, 30, 30, 32)        896       
# ______________________________________________________________
# max_pooling2d_4 (MaxPooling2 (None, 15, 15, 32)        0         
# ______________________________________________________________
# conv2d_7 (Conv2D)            (None, 13, 13, 64)        18496     
# ______________________________________________________________
# max_pooling2d_5 (MaxPooling2 (None, 6, 6, 64)          0         
# ______________________________________________________________
# conv2d_8 (Conv2D)            (None, 4, 4, 64)          36928     
# ______________________________________________________________
# flatten (Flatten)            (None, 1024)              0         
# ______________________________________________________________
# dense (Dense)                (None, 64)                65600     
# ______________________________________________________________
# dense_1 (Dense)              (None, 10)                650       
# ==============================================================
# Total params: 122,570
# Trainable params: 122,570
# Non-trainable params: 0
# ______________________________________________________________


# Training the model:
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

history = model.fit(train_images, train_labels, epochs=10, 
                    validation_data=(test_images, test_labels))

# Evaluating the model:
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print(test_acc)
