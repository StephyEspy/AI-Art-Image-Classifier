#!/usr/bin/env python
# coding: utf-8

# In[ ]:


get_ipython().system('pip install opendatasets')


# In[ ]:


get_ipython().system('pip install pandas')


# In[ ]:


import opendatasets as od
import pandas

od.download(
	"https://www.kaggle.com/datasets/ravidussilva/real-ai-art")


# In[ ]:


import os

os.listdir('real-ai-art/Real_AI_SD_LD_Dataset/train')
data_dir = 'real-ai-art/Real_AI_SD_LD_Dataset'

# Define the training paths
train_dir = os.path.join(data_dir, 'train')

# List all directories in the train directory
all_directories = os.listdir(train_dir)

# Initialize lists to store directories for human-drawn and AI-generated images
train_human = []
train_ai = []

# Loop through all directories
for dir in all_directories:
    # Check if the directory represents human-drawn images
    if not dir.startswith('AI_'):
        train_human.append(os.path.join(train_dir, dir))
    # Check if the directory represents AI-generated images
    else:
        train_ai.append(os.path.join(train_dir, dir))

# Print the lists of directories
print("Test directories containing human-drawn images:")
for idx, dir in enumerate(train_human):
    print(f"{idx}. {dir}")

print("\nTest directories containing AI-generated images:")
for idx, dir in enumerate(train_ai):
    print(f"{idx}. {dir}")


# In[ ]:


# Define the test paths
test_dir = os.path.join(data_dir, 'test')

# List all directories in the test directory
all_directories = os.listdir(test_dir)

# Initialize lists to store directories for human-drawn and AI-generated images
test_human = []
test_ai = []

# Loop through all directories
for dir in all_directories:
    # Check if the directory represents human-drawn images
    if not dir.startswith('AI_'):
        test_human.append(os.path.join(test_dir, dir))
    # Check if the directory represents AI-generated images
    else:
        test_ai.append(os.path.join(test_dir, dir))

# Print the lists of directories
print("Test directories containing human-drawn images:")
for idx, dir in enumerate(test_human):
    print(f"{idx}. {dir}")

print("\nTest directories containing AI-generated images:")
for idx, dir in enumerate(test_ai):
    print(f"{idx}. {dir}")


# In[ ]:


get_ipython().system('pip install keras_tuner')


# In[ ]:


import os
import random
from matplotlib import pyplot as plt
import cv2

import numpy as np
import pandas as pd

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.utils import image_dataset_from_directory
from keras.models import Sequential
from keras.layers import Rescaling, Conv2D, MaxPooling2D, Flatten, Dense, BatchNormalization, Activation, Dropout
from keras.metrics import Precision, Recall

import keras_tuner as kt
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping


# In[ ]:


# Plot k-number of images from the dataset
def plot_im(directory, k):
    files = os.listdir(directory)
    im = random.choices(files, k=k)

    fig = plt.figure()

    for i in range(k):
        im_i_path = os.path.join(directory, im[i])  # File path
        im_i = cv2.imread(im_i_path)

        # Add subplot
        ax = fig.add_subplot(int(np.sqrt(k)), int(np.sqrt(k)), i + 1)

        # Plot image
        ax.imshow(im_i)
        ax.axis('off')

        # Display filename below the image
        ax.set_title(im[i], fontsize = 8, pad = 2)

    plt.tight_layout()  # Adjust layout
    plt.show()


# In[ ]:


# Visualize random images from train_human. Catagory is sorted in order of output in cell 2
real_im = plot_im(directory = train_human[7], k = 9)
plt.show()


# In[ ]:


# Visualize random images from train_ai. Catagory is sorted in order of output in cell 2
ai_im = plot_im(directory = train_ai[4], k = 9)
plt.show()


# In[ ]:


# Initialize lists to store file paths and labels
paths = []
labels = []

# Initialize an empty DataFrame for train_data
train_data = pd.DataFrame(columns=['filepath', 'label'])

# Label files under train_human as "human"
for dir in train_human:
    for file in os.listdir(dir):
        filepath = os.path.join(dir, file)
        paths.append(filepath)
        labels.append("human")

# Label files under train_ai as "AI"
for dir in train_ai:
    for file in os.listdir(dir):
        filepath = os.path.join(dir, file)
        paths.append(filepath)
        labels.append("AI")

# Create a DataFrame with file paths and labels
data = pd.DataFrame({'filepath': paths, 'label': labels})

# Concatenate data with train_data
train_data = pd.concat([train_data, data], ignore_index=True)


# In[ ]:


# Display the first few rows of the train_data DataFrame
print(train_data.head())


# In[ ]:


# Count the number of files under each label
file_counts = train_data['label'].value_counts()

# Print the counts
print("Number of files under each label:")
print(file_counts)


# In[ ]:


import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
import os

# Define the augmentation parameters
humandata = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    shear_range=0.15,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode="nearest"
)

# Count images for each label
ai_count = train_data['label'].value_counts().get('AI', 0)
human_count = train_data['label'].value_counts().get('human', 0)

# Number of human augmented images needed to balance
augment_count = ai_count - human_count

# Only augment if there are fewer human images
if augment_count > 0:
    human_images = train_data[train_data['label'] == 'human']['filepath']
    augmented_paths, augmented_labels = [], []

    # Directory to save augmented images
    augmented_dir = "augmented_human_images/"
    os.makedirs(augmented_dir, exist_ok=True)

    # Generate augmented images across all human images
    for path in human_images:
        if augment_count <= 0:  # Stop when required augmentation is reached
            break

        image = tf.keras.preprocessing.image.load_img(path)
        image_array = tf.keras.preprocessing.image.img_to_array(image)
        image_array = image_array.reshape((1,) + image_array.shape)  # Reshape for batch

        # Generate augmented images and add them to path and label lists
        i = 0
        for batch in humandata.flow(image_array, batch_size=1):
            if augment_count <= 0:  # Stop when required augmentation is reached
                break

            # Define a unique filename for each augmented image
            unique_filename = f"{augmented_dir}augmented_{os.path.basename(path).split('.')[0]}_{i}.jpg"

            # Save augmented image
            aug_image = tf.keras.preprocessing.image.array_to_img(batch[0], scale=True)
            aug_image.save(unique_filename)

            # Append new paths and labels
            augmented_paths.append(unique_filename)
            augmented_labels.append('human')

            i += 1
            augment_count -= 1  # Decrement the remaining images to augment

    # Add augmented data to train_data
    augmented_data = pd.DataFrame({'filepath': augmented_paths, 'label': augmented_labels})
    train_data = pd.concat([train_data, augmented_data], ignore_index=True)

train_data


# In[ ]:


# Count the number of files under each label
file_counts = train_data['label'].value_counts()

# Print the counts
print("Number of files under each label:")
print(file_counts)


# In[ ]:


import matplotlib.pyplot as plt
import random
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# Directory where augmented images are saved
augmented_dir = "augmented_human_images/"

# Load a few random augmented images
augmented_images = random.sample(os.listdir(augmented_dir), 9)  # Display 9 images for example

# Plot the images in a 3x3 grid
plt.figure(figsize=(10, 10))
for i, img_name in enumerate(augmented_images):
    img_path = os.path.join(augmented_dir, img_name)
    img = load_img(img_path)  # Load image
    img_array = img_to_array(img)  # Convert to array

    plt.subplot(3, 3, i + 1)
    plt.imshow(img_array.astype("uint8"))
    plt.axis('off')
    plt.title("Augmented Image")

plt.tight_layout()
plt.show()


# In[ ]:


# Initialize lists to store file paths and labels
paths = []
labels = []

# Initialize an empty DataFrame for test_data
test_data = pd.DataFrame(columns=['filepath', 'label'])

# Label files under test_human as "human"
for dir in test_human:
    for file in os.listdir(dir):
        filepath = os.path.join(dir, file)
        paths.append(filepath)
        labels.append("human")

# Label files under test_ai as "AI"
for dir in test_ai:
    for file in os.listdir(dir):
        filepath = os.path.join(dir, file)
        paths.append(filepath)
        labels.append("AI")

# Create a DataFrame with file paths and labels
data = pd.DataFrame({'filepath': paths, 'label': labels})

# Concatenate data with test_data
test_data = pd.concat([test_data, data], ignore_index = True)


# In[ ]:


# Display the first few rows of the test_data DataFrame
print(test_data.head())

# Count the number of files under each label
file_counts = test_data['label'].value_counts()

# Print the counts
print("\nNumber of files under each label:")
print(file_counts)


# In[ ]:


training_generator = ImageDataGenerator(rescale=1./255,   # to normalize pixel value
                                       # rotation_range=7, # it will apply rotations to the image
                                       # horizontal_flip=True, # it will flip image horizontally
                                       # zoom_range=0.2  # it will increase and decrease zoom by 0.2x
                                       )


train_dataset = training_generator.flow_from_dataframe(
    dataframe=train_data,
    x_col='filepath',  # Column containing file paths
    y_col='label',     # Column containing labels
    target_size=(224, 224),
    batch_size=64,
    class_mode='binary',
    shuffle=True
)


# In[ ]:


train_dataset.class_indices


# In[ ]:


test_generator = ImageDataGenerator(rescale=1./255)
test_dataset = test_generator.flow_from_dataframe(  dataframe=test_data,
                                                    x_col='filepath',  # Column containing file paths
                                                    y_col='label',     # Column containing labels
                                                    target_size = (224, 224),
                                                    batch_size = 1,    # 1 image at a time to evaluate the NN
                                                    class_mode = 'binary',
                                                    shuffle = False)   # to associate the prediction with expected output


# In[ ]:


import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define image size and batch size
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

# Load ResNet50 with pre-trained weights, without the top layer
base_model = tf.keras.applications.ResNet50(weights = "imagenet", include_top=False, input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

# Freeze the base model
base_model.trainable = False

# Add new layers on top for binary classification
model = models.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(2, activation="softmax")  # Two classes: AI vs. non-AI
])

# Compile the model
model.compile(optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])


# In[ ]:


epochs = 5
history = model.fit(
    train_dataset,
    epochs=epochs,
    validation_data=test_dataset
)


# In[ ]:


model.save("my_model.keras")


# In[ ]:


# Fine-tuning: Unfreeze some layers of the base model and retrain with a lower learning rate
base_model.trainable = True
for layer in base_model.layers[:-10]:  # Freeze all layers except the last 10
    layer.trainable = False

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
              loss="sparse_categorical_crossentropy",
              metrics=["accuracy"])
# Continue training (fine-tuning)
fine_tune_epochs = 3
total_epochs = epochs + fine_tune_epochs

history_fine = model.fit(
    train_dataset,
    epochs=total_epochs,
    initial_epoch=history.epoch[-1],
    validation_data=test_dataset
)


# In[ ]:


# Plot the accuracies
import matplotlib.pyplot as plt

# Combine training and fine-tuning histories
train_acc = history.history['accuracy'] + history_fine.history['accuracy']
val_acc = history.history['val_accuracy'] + history_fine.history['val_accuracy']

# Plot the accuracies
plt.figure(figsize=(8, 6))
plt.plot(range(1, len(train_acc) + 1), train_acc, label='Training Accuracy')
plt.plot(range(1, len(val_acc) + 1), val_acc, label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.grid(True)
plt.show()


# In[ ]:


# Plot the confusion matrix
from sklearn import metrics
import matplotlib.pyplot as plt

def plot_confusion_matrix(actual, predicted, class_names):
    fig, ax = plt.subplots(figsize=(10,10))

    confusion_matrix = metrics.confusion_matrix(actual, predicted)

    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = class_names)

    cm_display.plot(ax=ax)
    ax.set_xticklabels(class_names, rotation=60)
    plt.show()


def get_predictions_and_labels(model, test_dataset):
    all_preds = []
    all_labels = []

    for batch in test_dataset:
        images, labels = batch  # Assuming dataset is in (image, label) format
        preds = model.predict(images, verbose=0)  # Get predictions

        all_preds.extend(np.argmax(preds, axis=1))  # Convert logits to class indices
        all_labels.extend(labels)  # Convert labels to NumPy array

    return np.array(all_labels), np.array(all_preds)


# After training the model, get predictions and true labels
actual, predicted = get_predictions_and_labels(model, test_dataset)

# Plot the confusion matrix
classes = ["human", "AI"] # This is just the list of classes, so AI-art v human-art in our case
plot_confusion_matrix(actual, predicted, classes)

# Adjust layout to fit nicely
plt.tight_layout()


# In[ ]:


# EfficientNetB0 to compare the pre-trained model
from tensorflow.keras.applications import EfficientNetB0
efficient_base = EfficientNetB0(weights = "imagenet", include_top=False, input_shape=(224, 224, 3))
efficient_base.trainable = False # Freeze the pre-trained layers

efficient_model = models.Sequential([
    efficient_base,
    layers.GlobalAveragePooling2D(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),
    layers.Dense(2, activation="softmax")  # Two classes: AI vs. non-AI
])

efficient_model.compile(
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.001),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Train EfficientNetB0
efficient_history = efficient_model.fit(
    train_dataset,
    epochs = 5,
    validation_data = test_dataset
)

# Unfreeze the base model
efficient_base.trainable = True
for layer in efficient_base.layers[:-10]:  # Freeze all layers except the last 10
    layer.trainable = False

efficient_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# Continue training
fine_tune_history = efficient_model.fit(
    train_dataset,
    epochs = 3,
    initial_epoch = efficient_history.epoch[-1],
    validation_data = test_dataset
)

resnet_train_acc = history.history['accuracy'] + history_fine.history['accuracy']
resnet_val_acc = history.history['val_accuracy'] + history_fine.history['val_accuracy']

efficient_train_acc = efficient_history.history['accuracy'] + fine_tune_history.history['accuracy']
efficient_val_acc = efficient_history.history['val_accuracy'] + fine_tune_history.history['val_accuracy']

plt.figure(figsize = (10, 6))
plt.plot(resnet_train_acc, label = "ResNet50 Training Accuracy")
plt.plot(resnet_val_acc, label = "ResNet50 Validation Accuracy")
plt.plot(efficient_train_acc, label = "EfficientNetB0 Training Accuracy")
plt.plot(efficient_val_acc, label = 'EfficientNetB0 Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')
plt.show()

