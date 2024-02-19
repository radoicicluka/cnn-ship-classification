# In [1]:
import tensorflow as tf
import logging
tf.get_logger().setLevel(logging.ERROR)

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

path = 'new_images/'
img_size = (160, 160)
batch_size = 64

train_file = pd.read_csv('archive/train/train.csv')

cnt = [train_file['category'].value_counts()[4],
        train_file['category'].value_counts()[3],
        train_file['category'].value_counts()[2],
        train_file['category'].value_counts()[5],
        train_file['category'].value_counts()[1]]
plt.pie(cnt, labels=[
                    'Cruise - '+ str(cnt[0]), 
                    'Carrier - '+ str(cnt[1]), 
                    'Military - '+ str(cnt[2]), 
                    'Tankers - '+ str(cnt[3]),
                    'Cargo - '+ str(cnt[4]) 
                    ])

# In[2]:
from keras.utils import image_dataset_from_directory

train_data = image_dataset_from_directory(path, 
                                          subset='training', 
                                          validation_split=0.2, 
                                          image_size=img_size, 
                                          batch_size=batch_size, 
                                          seed=221)
val_data = image_dataset_from_directory(path, 
                                        subset='validation', 
                                        validation_split=0.2, 
                                        image_size=img_size, 
                                        batch_size=batch_size, 
                                        seed=221)

classes = train_data.class_names
print(classes)

weights = {0: 6252/(5*2120), 
           1: 6252./(5*916), 
           2: 6252./(5*832), 
           3: 6252./(5*1167), 
           4: 6252./(5*1217)
           }
# cargo, carrier, cruise, military, tanker
# In[3]:
N = 10
plt.figure()
for img, lab in train_data.take(1):
    for i in range(N):
        plt.subplot(2, int(N/2), i + 1)
        plt.imshow(img[i].numpy().astype('uint8'))
        plt.title(classes[lab[i]])
        plt.axis('off')
plt.show()

# In[4]:
from keras import Sequential
from keras.optimizers.legacy import Adam  # legacy verzija zbog rada na ARM procesoru
from keras.losses import SparseCategoricalCrossentropy
from keras import layers

data_augmentation = Sequential([
    layers.RandomFlip('horizontal', input_shape=(img_size[0], img_size[1], 3), seed=2), 
    layers.RandomZoom(0.1, seed=2),
    layers.RandomRotation(0.1, seed=2),
    layers.RandomBrightness(0.5, seed=2),
    layers.RandomContrast(0.25, seed=2)
])

N = 10
plt.figure()
plt.axis('off')
for img, lab in train_data.take(1):
    plt.title(classes[lab[0]])
    for i in range(N):
        aug_img = data_augmentation(img)
        plt.subplot(2, int(N/2), i + 1)
        plt.imshow(aug_img[0].numpy().astype('uint8'))
        plt.axis('off')
plt.show()

# In[5]:

num_classes = len(classes)

model = Sequential([
    data_augmentation, 
    layers.Rescaling(1./255, input_shape=(img_size[0], img_size[1], 3)), 
    layers.Conv2D(16, 3, padding='same', strides=1, activation='relu'), 
    layers.MaxPooling2D(),  
    layers.Conv2D(32, 3, padding='same', strides=1, activation='relu'), 
    layers.MaxPooling2D(),  
    layers.Conv2D(64, 3, padding='same', strides=1, activation='relu'), 
    layers.MaxPooling2D(),  
    layers.Conv2D(128, 3, padding='same', strides=1, activation='relu'), 
    layers.MaxPooling2D(),
    layers.Dropout(0.2), 
    layers.Flatten(), 
    layers.Dense(128, activation='relu'), 
    layers.Dense(num_classes, activation='softmax') 
])

model.summary()

model.compile(Adam(learning_rate=0.001), loss=SparseCategoricalCrossentropy(), metrics='accuracy')

# In[6]:
from keras.callbacks import EarlyStopping
es = EarlyStopping(monitor='loss', mode='min', patience=50, restore_best_weights=True, verbose=1)
history = model.fit(train_data, 
                    epochs=1000,
                    validation_data=val_data, 
                    verbose=1, 
                    class_weight=weights, 
                    callbacks=[es])

# In[7]:

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

plt.figure()
plt.subplot(121)
plt.plot(acc)
plt.plot(val_acc)
plt.title('Accuracy')
plt.subplot(122)
plt.plot(loss)
plt.plot(val_loss)
plt.title('Loss')
plt.show()

# In[8]:
labels_val = np.array([])
pred_val = np.array([])
for img, lab in val_data:
    labels_val = np.append(labels_val, lab)
    pred_val = np.append(pred_val, np.argmax(model.predict(img, verbose=0), axis=1))


# In[9]:
from sklearn.metrics import accuracy_score
print('Accuracy: ' + str(100*accuracy_score(labels_val, pred_val)) + '%')

labels_train = np.array([])
pred_train = np.array([])
for img, lab in train_data:
    labels_train = np.append(labels_train, lab)
    pred_train = np.append(pred_train, np.argmax(model.predict(img, verbose=0), axis=1))


from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
cm_val = confusion_matrix(labels_val, pred_val, normalize='true')
cmDisplayVal = ConfusionMatrixDisplay(confusion_matrix=cm_val, display_labels=classes)
cmDisplayVal.plot()
plt.show()

cm_train = confusion_matrix(labels_train, pred_train, normalize='true')
cmDisplayTrain = ConfusionMatrixDisplay(confusion_matrix=cm_train, display_labels=classes)
cmDisplayTrain.plot()
plt.show()

model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
