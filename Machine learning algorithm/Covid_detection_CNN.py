# -*- coding: utf-8 -*-

from keras.layers import Conv2D, Dense, Flatten, MaxPooling2D, Dropout
from keras.models import Sequential

classifier = Sequential()

classifier.add(Conv2D(16, (5, 5), input_shape = (128, 128, 3), activation = 'relu', padding = 'same'))
classifier.add(MaxPooling2D(pool_size = (2, 2), padding = 'same'))

classifier.add(Conv2D(8, (3, 3), activation = 'relu', padding = 'same'))
classifier.add(MaxPooling2D(pool_size = (2, 2), padding = 'same'))

classifier.add(Conv2D(8, (3, 3), activation = 'relu', padding = 'same'))
classifier.add(MaxPooling2D(pool_size = (2, 2), padding = 'same'))

classifier.add(Flatten())

classifier.add(Dense(units = 16, activation = 'relu'))
classifier.add(Dropout(rate=0.1))

classifier.add(Dense(units = 16, activation = 'relu'))
classifier.add(Dropout(rate = 0.1))

classifier.add(Dense(units = 2, activation = 'softmax'))

classifier.compile(optimizer = 'RMSprop', loss = 'categorical_crossentropy', metrics=['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator()
test_datagen = ImageDataGenerator()
training_set = train_datagen.flow_from_directory('train',
                                                 target_size = (128, 128),
                                                 batch_size = 16,
                                                 class_mode = 'categorical')
test_set = test_datagen.flow_from_directory('test',
                                            target_size = (128, 128),
                                            batch_size = 16,
                                            class_mode = 'categorical')

classifier.fit_generator(training_set,
                         steps_per_epoch = 277,
                         epochs = 3,
                         validation_data = test_set,
                         validation_steps = 46,
                         workers=16,
                         max_queue_size=10)

import numpy as np
from keras.preprocessing import image
test_image = image.load_img('single prediction/covid/covid_(8).jpg', target_size = (128, 128))
test_image = image.img_to_array(test_image)
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices
if result[0][0] > 0.5:
    prediction = "you have covid"
else:
    prediction = "you don't have covid"
print(prediction)

classifier.save("model progress")