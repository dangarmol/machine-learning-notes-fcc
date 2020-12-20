from keras.datasets import imdb
from keras.preprocessing import sequence
import keras
import tensorflow as tf
import os
import numpy as np

VOCAB_SIZE = 88584

MAXLEN = 250
BATCH_SIZE = 64

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words = VOCAB_SIZE)

print(train_data[1])  # Each review is a list of numbers

train_data = sequence.pad_sequences(train_data, MAXLEN)  # Making every review the same length by trimming or padding.
test_data = sequence.pad_sequences(test_data, MAXLEN)


# Creating the model
model = tf.keras.Sequential([
    tf.keras.layers.Embedding(VOCAB_SIZE, 32),
    tf.keras.layers.LSTM(32),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

model.summary()
# Model: "sequential"
# _________________________________________________________________
# Layer (type)                 Output Shape              Param #   
# =================================================================
# embedding (Embedding)        (None, None, 32)          2834688   
# _________________________________________________________________
# lstm (LSTM)                  (None, 32)                8320      
# _________________________________________________________________
# dense (Dense)                (None, 1)                 33        
# =================================================================
# Total params: 2,843,041
# Trainable params: 2,843,041
# Non-trainable params: 0
# _________________________________________________________________


# Compiling and training the model:
model.compile(loss="binary_crossentropy",optimizer="rmsprop",metrics=['acc'])

history = model.fit(train_data, train_labels, epochs=10, validation_split=0.2)


# Evaluating the model:
results = model.evaluate(test_data, test_labels)
print(results)


# Encoding, decoding and making predictions:
word_index = imdb.get_word_index()

# Encoding the text
def encode_text(text):
	tokens = keras.preprocessing.text.text_to_word_sequence(text)
	tokens = [word_index[word] if word in word_index else 0 for word in tokens]
	return sequence.pad_sequences([tokens], MAXLEN)[0]

text = "that movie was just amazing, so amazing"
encoded = encode_text(text)
print(encoded)

reverse_word_index = {value: key for (key, value) in word_index.items()}

# Decoding the text
def decode_integers(integers):
    PAD = 0
    text = ""
    for num in integers:
    	if num != PAD:
        text += reverse_word_index[num] + " "

    return text[:-1]
  
print(decode_integers(encoded))


# Making predictions:
def predict(text):
  encoded_text = encode_text(text)
  pred = np.zeros((1,250))
  pred[0] = encoded_text
  result = model.predict(pred) 
  print(result[0])

positive_review = "That movie was! really loved it and would great watch it again because it was amazingly great"
predict(positive_review)  # Outputs [0.82867914]

negative_review = "that movie really sucked. I hated it and wouldn't watch it again. Was one of the worst things I've ever watched"
predict(negative_review)  # Outputs [0.2008027]

