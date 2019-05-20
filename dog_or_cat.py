import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, LearningRateScheduler, ModelCheckpoint, TensorBoard
from sklearn.metrics import confusion_matrix

#Set parameters
image_size = 64
batches = 32


#Initialize a sequential neural network
model = Sequential()

#Add convolution layer with 32 filters at 3x3 size, 64x64 resolution and RGB
model.add(Conv2D(32,
                (3, 3),
                input_shape = (image_size, image_size, 3),
                activation = 'relu'))

#Adding max pooling layer with size 2x2
model.add(MaxPooling2D(pool_size = (2,2)))


model.add(Conv2D(64,
                (3, 3),
                input_shape = (image_size, image_size, 3),
                activation = 'relu'))

model.add(MaxPooling2D(pool_size = (2,2)))


model.add(Conv2D(128,
                (3, 3),
                input_shape = (image_size, image_size, 3),
                activation = 'relu'))

model.add(MaxPooling2D(pool_size = (2,2)))

#Flatten the pooled array
model.add(Flatten())

#Add layer with 256 nodes
model.add(Dense(units = 256, activation = 'relu'))
#Add dropout rate of 0.4 to avoid overfitting
model.add(Dropout(0.4))
#Add output layer with one node
model.add(Dense(units = 1, activation = 'sigmoid'))

#Compile the CNN with stochastic gradient decent, log-loss and performance metric
model.compile(optimizer = 'adam',
            loss = 'binary_crossentropy',
            metrics = ['accuracy'])

#Train the data generator on different augmentations of the pictures
train_datagen = ImageDataGenerator(rescale = 1./255,
                                shear_range = 0.2,
                                zoom_range = 0.2,
                                horizontal_flip = True)

validation_datagen = ImageDataGenerator(rescale = 1./255)

test_datagen = ImageDataGenerator(rescale = 1./255)


training_set = train_datagen.flow_from_directory('training',
                                                target_size = (image_size, image_size),
                                                batch_size = batches,
                                                class_mode = 'binary')

validation_set = validation_datagen.flow_from_directory('validation',
                                                        target_size = (image_size, image_size),
                                                        batch_size = batches,
                                                        class_mode = 'binary')

test_set = test_datagen.flow_from_directory('test_folder',
                                            target_size = (image_size, image_size),
                                            batch_size = 1,
                                            class_mode = 'binary',
                                            shuffle = False)


#Set checkpoint and early stopping so the epochs will stop when it has 3 epochs in a row with no performance increase on the validation set
checkpoint = ModelCheckpoint("validation_accuracy", monitor='val_acc', verbose=1, save_best_only=True, save_weights_only=True, mode='auto', period=1)
early = EarlyStopping(monitor='val_acc', min_delta=0, patience=3, verbose=1, mode='auto',restore_best_weights=True)

#Run the training set with validation and save it to a variable to plot it
history = model.fit_generator(training_set,
                    steps_per_epoch = len(training_set.filenames) // batches,
                    epochs = 30,
                    validation_data = validation_set,
                    validation_steps = len(validation_set.filenames) // batches,
                    callbacks = [checkpoint, early])

#Plot accuracy of training and validation
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')
plt.show()

prob = model.predict_generator(test_set, 6000)

y_true = np.array([0] * 3000 + [1] * 3000)
y_pred = prob > 0.5

a = confusion_matrix(y_true, y_pred)
print("True Positive    False Positive")
print("False Negative   True Negative")
print(a)
