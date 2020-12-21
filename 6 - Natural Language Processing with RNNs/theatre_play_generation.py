from keras.preprocessing import sequence
import keras
import tensorflow as tf
import os
import numpy as np

# If we want to load a dataset of the Shakespeare script:
path_to_file = tf.keras.utils.get_file('shakespeare.txt', 'https://storage.googleapis.com/download.tensorflow.org/data/shakespeare.txt')

# If we want to load our own dataset:
from google.colab import files
path_to_file = list(files.upload().keys())[0]


# Reading the contents...
# Read, then decode for py2 compat.
text = open(path_to_file, 'rb').read().decode(encoding='utf-8')
# length of text is the number of characters in it
print ('Length of text: {} characters'.format(len(text)))

# Take a look at the first 250 characters in text
print(text[:250])


# Encoding...
vocab = sorted(set(text))
# Creating a mapping from unique characters to indices
char2idx = {u:i for i, u in enumerate(vocab)}
idx2char = np.array(vocab)

def text_to_int(text):
	return np.array([char2idx[c] for c in text])

text_as_int = text_to_int(text)

# lets look at how part of our text is encoded
print("Text:", text[:13])
print("Encoded:", text_to_int(text[:13]))

# Function that can convert our numeric values to text
def int_to_text(ints):
	try:
		ints = ints.numpy()
	except:
		pass
	return ''.join(idx2char[ints])

print(int_to_text(text_as_int[:13]))


# Creating Training Examples...
seq_length = 100  # length of sequence for a training example
examples_per_epoch = len(text)//(seq_length+1)

# Create training examples / targets
char_dataset = tf.data.Dataset.from_tensor_slices(text_as_int)

sequences = char_dataset.batch(seq_length+1, drop_remainder=True)

def split_input_target(chunk):  # for the example: hello
	input_text = chunk[:-1]  # hell
	target_text = chunk[1:]  # ello
	return input_text, target_text  # hell, ello

dataset = sequences.map(split_input_target)  # we use map to apply the above function to every entry

for x, y in dataset.take(2):
	print("\n\nEXAMPLE\n")
	print("INPUT")
	print(int_to_text(x))
	print("\nOUTPUT")
	print(int_to_text(y))

# Training batches...
BATCH_SIZE = 64
VOCAB_SIZE = len(vocab)  # vocab is number of unique characters
EMBEDDING_DIM = 256
RNN_UNITS = 1024

# Buffer size to shuffle the dataset
# (TF data is designed to work with possibly infinite sequences,
# so it doesn't attempt to shuffle the entire sequence in memory. Instead,
# it maintains a buffer in which it shuffles elements).
BUFFER_SIZE = 10000


# Building the model...
def build_model(vocab_size, embedding_dim, rnn_units, batch_size):
  model = tf.keras.Sequential([
    tf.keras.layers.Embedding(vocab_size, embedding_dim,
                              batch_input_shape=[batch_size, None]),
    tf.keras.layers.LSTM(rnn_units,
                        return_sequences=True,  # Return sequences needs to be true so that we see the output on each step (letter by letter)
                        stateful=True,
                        recurrent_initializer='glorot_uniform'),
    tf.keras.layers.Dense(vocab_size)
  ])
  return model

model = build_model(VOCAB_SIZE,EMBEDDING_DIM, RNN_UNITS, BATCH_SIZE)
model.summary()

# Creating a loss function
for input_example_batch, target_example_batch in data.take(1):
	example_batch_predictions = model(input_example_batch)  # ask our model for a prediction on our first batch of training data (64 entries)
	print(example_batch_predictions.shape, "# (batch_size, sequence_length, vocab_size)")  # print out the output shape

data = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE, drop_remainder=True)

# lets examine one prediction
pred = example_batch_predictions[0]
print(len(pred))
print(pred)
# notice this is a 2d array of length 100, where each interior array is the prediction for the next character at each time step

# and finally well look at a prediction at the first timestep
time_pred = pred[0]
print(len(time_pred))
print(time_pred)
# and of course its 65 values representing the probabillity of each character occuring next

# If we want to determine the predicted character we need to sample the output distribution (pick a value based on probability)
# Sampling is not the same as simply picking up the character with the highest probability! Doing this could cause some unexpected behaviour
sampled_indices = tf.random.categorical(pred, num_samples=1)

# now we can reshape that array and convert all the integers to numbers to see the actual characters
sampled_indices = np.reshape(sampled_indices, (1, -1))[0]
predicted_chars = int_to_text(sampled_indices)

predicted_chars  # and this is what the model predicted for training sequence 1 (before training it on real data, just random weights and biases)

def loss(labels, logits):
	return tf.keras.losses.sparse_categorical_crossentropy(labels, logits, from_logits=True)


# Compiling the model...
model.compile(optimizer='adam', loss=loss)


# Creating checkpoints...
# This will allow us to load our model from a checkpoint and continue training it.
# Directory where the checkpoints will be saved
checkpoint_dir = './training_checkpoints'
# Name of the checkpoint files
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt_{epoch}")

checkpoint_callback=tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_prefix,
    save_weights_only=True)


# Training the model, recommended to use GPU
history = model.fit(data, epochs=50, callbacks=[checkpoint_callback])

# Loading the model
model = build_model(VOCAB_SIZE, EMBEDDING_DIM, RNN_UNITS, batch_size=1)

# Find latest checkpoint
model.load_weights(tf.train.latest_checkpoint(checkpoint_dir))
model.build(tf.TensorShape([1, None]))

# Loading a checkpoint (if there are any)
checkpoint_num = 10
model.load_weights(tf.train.load_checkpoint("./training_checkpoints/ckpt_" + str(checkpoint_num)))
model.build(tf.TensorShape([1, None]))


# Function to use for text prediction:
def generate_text(model, start_string):
  # Evaluation step (generating text using the learned model)

  # Number of characters to generate
  num_generate = 800

  # Converting our start string to numbers (vectorizing)
  input_eval = [char2idx[s] for s in start_string]
  input_eval = tf.expand_dims(input_eval, 0)

  # Empty string to store our results
  text_generated = []

  # Low temperatures results in more predictable text.
  # Higher temperatures results in more surprising text.
  # Experiment to find the best setting.
  temperature = 1.0

  # Here batch size == 1
  model.reset_states()
  for i in range(num_generate):
      predictions = model(input_eval)
      # remove the batch dimension
    
      predictions = tf.squeeze(predictions, 0)

      # using a categorical distribution to predict the character returned by the model
      predictions = predictions / temperature
      predicted_id = tf.random.categorical(predictions, num_samples=1)[-1,0].numpy()

      # We pass the predicted character as the next input to the model
      # along with the previous hidden state
      input_eval = tf.expand_dims([predicted_id], 0)

      text_generated.append(idx2char[predicted_id])

  return (start_string + ''.join(text_generated))


# Trying out our model by asking the user for input as a starting string and printing the output!
inp = input("Type a starting string: ")
print(generate_text(model, inp))