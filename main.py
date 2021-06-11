# TensorFlow and tf.keras
import tensorflow as tf

# Helper libraries
import numpy as np
import matplotlib.pyplot as plt

# SETUP
fashion_mnist = tf.keras.datasets.fashion_mnist  # grab library from tensorflow
(train_images, train_labels), (test_images,
                               test_labels) = fashion_mnist.load_data()  # 2  28 x 28 matrix arrays for pixel value of each image and a label
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']  # to associate the label above to a word


# PREPROCESSING

# reduce size of images from 0-255 pixels to 0-1
train_images = train_images / 255.0
test_images = test_images / 255.0

# PROCESSING

# configure layers
model = tf.keras.Sequential([
    # reformat the data to a single matrix array of values
    tf.keras.layers.Flatten(input_shape=(28, 28)),
    # 128 node layer
    tf.keras.layers.Dense(128, activation='relu'),
    # return logit array with score 0 - 10 that corresponds to class
    tf.keras.layers.Dense(10)
])

# COMPILE
model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(
                  from_logits=True),
              metrics=['accuracy'])

# TRAIN
model.fit(train_images, train_labels, epochs=10)

# VALIDATE / TEST
test_loss, test_acc = model.evaluate(test_images,  test_labels, verbose=2)
print('\nTest accuracy:', test_acc)

# PREDICTION
probability_model = tf.keras.Sequential(
    [model, tf.keras.layers.Softmax()])  # convert logit to percentages
predictions = probability_model.predict(test_images)


def plot_image(i, predictions_array, true_label, img):
    true_label, img = true_label[i], img[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img, cmap=plt.cm.binary)

    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label:
        color = 'blue'
    else:
        color = 'red'

    plt.xlabel("{} {:2.0f}% ({})".format(class_names[predicted_label],
                                         100*np.max(predictions_array),
                                         class_names[true_label]),
               color=color)


def plot_value_array(i, predictions_array, true_label):
    true_label = true_label[i]
    plt.grid(False)
    plt.xticks(range(10))
    plt.yticks([])
    thisplot = plt.bar(range(10), predictions_array, color="#777777")
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)

    thisplot[predicted_label].set_color('red')
    thisplot[true_label].set_color('blue')


i = 6
plt.figure(figsize=(6, 3))
plt.subplot(1, 2, 1)
plot_image(i, predictions[i], test_labels, test_images)
plt.subplot(1, 2, 2)
plot_value_array(i, predictions[i],  test_labels)
plt.show()

# CLEAN UP
# request enter key to stop terminal closing
input()
