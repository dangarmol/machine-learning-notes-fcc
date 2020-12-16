# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt


fashion_mnist = keras.datasets.fashion_mnist  # load dataset
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()  # split into testing and training

print(train_images.shape)  # (60000, 28, 28) = 60000 images made of 28 x 28 pixels
print(train_images[0,23,23])  # We can access single pixels of an image with this syntax, similar to "train_images[0][23][23]"
print(train_labels[:10])  # let's have a look at the first 10 training labels -> array([9, 0, 0, 3, 0, 2, 7, 2, 5, 5], dtype=uint8)

# Labels are numbers 0-9, so we create a list for reference
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# We can also draw some of the images to check them visually by calling this
'''
plt.figure()
plt.imshow(train_images[1], cmap="gray_r")
plt.colorbar()
plt.grid(False)
plt.show()
'''

# For the model to be trained efficiently, the input values should be between 0 and 1, since we will assign random initial values to the neurons in that range
train_images = train_images / 255.0
test_images = test_images / 255.0

# Keras sequential model represents a feed-forward NN, passing values left to right
# Each line represents a layer
# These arguments we are giving are called Hyperparameters and can affect the results of the training quite a lot!
model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28, 28)),  # input layer (1) - 784 neurons with the input values
    keras.layers.Dense(128, activation='relu'),  # hidden layer (2) - 128 neurons with ReLU and fully connected to the previous layer (dense)
    keras.layers.Dense(10, activation='softmax') # output layer (3) - 10 neurons for 10 possible outputs. Softmax creates a probability distribution of this layer between 0 and 1 adding up to 1
])

# Compiling the model with a cost function and other arguments
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# We pass the data, epochs and actually train the model here
model.fit(train_images, train_labels, epochs=10)

# Check the stats from the model
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=1) 
print('Test accuracy:', test_acc)


# We can make model predictions
predictions = model.predict(test_images)
# It will return an array of arrays with the probabilities of each element.

# We can get the element with the max probability
np.argmax(predictions[0])  # Prints "9"
# And compare to the actual label
test_labels[0]  # Prints "9"
