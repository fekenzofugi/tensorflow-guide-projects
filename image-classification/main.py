import tensorflow as tf 
from tensorflow.keras.datasets import fashion_mnist
import matplotlib.pyplot as plt
import random
import pandas as pd
import numpy as np
from utils import make_confusion_matrix
from utils import plot_random_image
from tensorflow.keras.utils import plot_model

"""
    Steps in modeling with TensorFlow
    1) Get data ready (turn into tensors)
    2) Build or pick a pretrained model (to suit your problem)
    3) Fit the model to the data and make a prediction
    4) Evaluate the model
    5) Improve through experimentation
    6) Save and reload your trained model
"""

# The data has already been sorted into training and test sets for us
(train_data, train_labels), (test_data, test_labels) = fashion_mnist.load_data()

# Show a training example
index_of_choice = 17

print(f"Training sample:\n{train_data[index_of_choice]}\n")
print(f"training label:\n{train_labels[index_of_choice]}\n")

# Check the shape of a single example
print(train_data[index_of_choice].shape, train_labels[index_of_choice].shape)

# Check out samples label
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# plt.imshow(train_data[index_of_choice])
# plt.title(class_names[train_labels[index_of_choice]])
# plt.show()

# Plot multiple random images 
# n = 4
# plt.figure(figsize=(7, 7))
# for i in range(n):
#     ax = plt.subplot(2, 2, i+1)
#     rand_index = random.choice(range(len(train_data)))
#     plt.imshow(train_data[rand_index])
#     plt.title(class_names[train_labels[rand_index]])
#     plt.axis(False)
# plt.show()

# Building a multi-class classification model
"""
    input-shape = 28x28 (shape of one image)
    output-shape = 10 (one per class of clothing)
    loss function = tf.keras.losses.CategoricalCrossentropy()
        if your labels are one-hot encoded, use CategoricalCrossenetropy()
        if your labels are integers, use SparseCategoricalCrossentropy()
    ouput layer activation = Softmax (not sigmoid)
"""

# Our data needs to be flattened (from 28, 28 to None, 784)
flatten_model = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(28, 28))
])
flatten_model.output_shape

# # Set random seed
# tf.random.set_seed(42)

# # Create the model
# model_11 = tf.keras.Sequential([
#     tf.keras.layers.Flatten(input_shape=(28, 28)),
#     tf.keras.layers.Dense(units=4, activation="relu"),
#     tf.keras.layers.Dense(units=4, activation="relu"),
#     tf.keras.layers.Dense(units=10, activation=tf.keras.activations.softmax)
# ])

# # Compile the model
# model_11.compile(
#     loss=tf.keras.losses.SparseCategoricalCrossentropy(),
#     optimizer=tf.keras.optimizers.Adam(),
#     metrics=["accuracy"]
# )

# # Fit the model
# non_norm_history = model_11.fit(
#     train_data, 
#     train_labels, 
#     epochs=10, 
#     validation_data=(test_data, test_labels)
# )

# # Check the model summary
# model_11.summary()

"""
1) Get data ready
    1. Turn all data into numbers
    2. Make sure all your tensors are in the right shape
    3. Scale features (normalize or standardize). 
        Neural networks prefer data to be scaled (or normalized), this means they like to have the numbers in the tensors they try to find patterns between 0 and 1.
"""

# Check the min and max values of the training data
print(train_data.min())
print(train_data.max())

# Normalize data
train_data_norm = train_data / 255.0
test_data_norm = test_data / 255.0

# Check the min and max values of the scaled training data
print(train_data_norm.min())
print(train_data_norm.max())

# Now our data is normalized, let's build a model to find patterns in it

# # Set random seed
# tf.random.set_seed(42)

# # Create a model (same as model_11)
# model_12 = tf.keras.Sequential([
#     tf.keras.layers.Flatten(input_shape=(28, 28)),
#     tf.keras.layers.Dense(4, activation="relu"),
#     tf.keras.layers.Dense(4, activation="relu"),
#     tf.keras.layers.Dense(10, activation="softmax")
# ])

# # Compile the model
# model_12.compile(
#     loss=tf.keras.losses.SparseCategoricalCrossentropy(),
#     optimizer=tf.keras.optimizers.Adam(),
#     metrics=["accuracy"]
# )

# # Fit the model
# norm_history = model_12.fit(
#     train_data_norm,
#     train_labels,
#     epochs=10,
#     validation_data=(test_data_norm, test_labels)
# )

# # Plot non-normalized data loss curves
# pd.DataFrame(non_norm_history.history).plot(title="Non-norrmalized data")
# # Plot normalized data loss curves
# pd.DataFrame(norm_history.history).plot(title="Normalized data")
# plt.show()

"""
    The same model with even "slightly different data produce dramatically different results. So when you're comparing models, it's important to make sure you're comparing them on the same criteria.
"""

# # Finding the ideal learning rate

# # set the random seed
# tf.random.set_seed(42)

# # Create the model
# model_13 = tf.keras.Sequential([
#     tf.keras.layers.Flatten(input_shape=(28, 28)),
#     tf.keras.layers.Dense(4, activation="relu"),
#     tf.keras.layers.Dense(4, activation="relu"),
#     tf.keras.layers.Dense(10, activation="softmax")
# ])

# # Compile the model
# model_13.compile(
#     loss=tf.keras.losses.SparseCategoricalCrossentropy(),
#     optimizer=tf.keras.optimizers.Adam(),
#     metrics=["accuracy"]
# )

# # Create the learning rate callback
# """
#     Start at 1e-3 and for each epoch increase the learning rate every epoch by 10 ** (epoch/20)
# """
# lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-3 * 10 **(epoch/20))

# # Fit the model
# find_lr_history = model_13.fit(
#     train_data_norm,
#     train_labels,
#     epochs=40,
#     validation_data=(test_data_norm, test_labels),
#     callbacks=[lr_scheduler]
# )

# # plot the learning rate decay curve
# lrs = 1e-3 * (10**(tf.range(40)/20))
# plt.semilogx(lrs, find_lr_history.history["loss"])
# plt.xlabel("Learning Rate")
# plt.ylabel("Loss") 
# plt.title("Finding the ideal learning rate")
# plt.show()

# Let's refit the model with the ideal learning rate

# Set random seed
tf.random.set_seed(42)

# Create the model
model_14 = tf.keras.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)), # input layer (we had to reshape 28x28 to 784)
  tf.keras.layers.Dense(4, activation="relu"),
  tf.keras.layers.Dense(4, activation="relu"),
  tf.keras.layers.Dense(10, activation="softmax") # output shape is 10, activation is softmax
])

# Compile the model
model_14.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(),
                 optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), # ideal learning rate (same as default)
                 metrics=["accuracy"])

# Fit the model
history = model_14.fit(train_data_norm,
                       train_labels,
                       epochs=20,
                       validation_data=(test_data_norm, test_labels))

# Evaluating ouru multi-class classification model
"""
    To evaluate our multi-class classification model we could:
    - Evaluate its performance using other classification metrics (such as confusion matrix)
    - Assess some of its predictions through visualizations
    - Improve its results (by training it for longer or changing the architecture)
    - Save and export it to use in an application
"""

# Make some predictions with our model

y_probs = model_14.predict(test_data_norm) # props is short for "prediction probabilities". 

# Remember to make predictions on the same kind of data your model was trained. (e.g. if your model was trained on normalized data, you'll want to make predictions on normalized data).

# View the first 5 predictions
print(y_probs[0])
print(class_names[tf.argmax(y_probs[0])])

# Convert all the prediction probabilities into integers
y_preds = y_probs.argmax(axis=1)

# View the first 10 prediction labels
print(y_preds[:10])

# Make a confusion matrix
make_confusion_matrix(
    y_true=test_labels,
    y_pred=y_preds,
    classes=class_names,
    figsize=(15, 15),
    text_size=10
)

"""
    Often when working with images and other forms of visual data, it's a good idea to visualize as much as possible to develop a further understanding of the data and the inputs and outputs of your models.
"""

# Check out a random image as well as its prediction
n = 4
for i in range(n):
    plot_random_image(
        model=model_14,
        images=test_data_norm,
        true_labels=test_labels,
        classes=class_names
    )


# Find the layers of our most recent model
print(model_14.layers)

# Extract a particular layer
print(model_14.layers[1])

# Get the patterns of a layer in our network 
weights, biases = model_14.layers[1].get_weights()

# Shapes
print(weights)
print(weights.shape)

# Bias and biases shapes
print(biases)
print(biases.shape)

print(model_14.summary())


"""
Every neuron has a bias vector. Each of these is paired with a weights matrix.

The bias vector gets initialized as zeros -> Dense layer.

The bias vector dictates how much the patterns within the corresponding weights matrix should influence the next layer.
"""

# Let's check out another way of viewing our deep learning models

# See the inputs and outputs of each layer
plot_model(model_14, show_shapes=True)