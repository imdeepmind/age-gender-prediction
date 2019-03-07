##################################
# STEP - 1: Importing Dependencies
##################################

# Importing dependencies
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.applications import MobileNetV2
from keras import optimizers
from keras.preprocessing.image import ImageDataGenerator

#####################
# STEP - 2: The Model
#####################

# Making the model
model = Sequential()

# For this model I'm using InceptionResNetV2
# I'll use imagenet weights here also
mobile = MobileNetV2(include_top=False,
                          weights="imagenet", 
                          input_shape=(128,128,3),
                          pooling="max")

# Adding the mobile model and configuting the output layer
model.add(mobile)
model.add(Dense(units=2, activation="softmax"))

# Compiling the model
model.compile(loss='binary_crossentropy', optimizer=optimizers.RMSprop(lr=2e-5), metrics=['accuracy'])

# Printing the summary of the model
print(model.summary())

############################
# STEP 3: Data Preprocessing
############################

# Here I'm using ImageDataGenerator class for data augmentation
train_datagen = ImageDataGenerator(rescale=1./255,shear_range=0.2,zoom_range=0.3)
test_datagen = ImageDataGenerator(rescale=1./255)

# Reading the training set
train_generator = train_datagen.flow_from_directory('dataset/gender/train',
                                                    target_size=(128, 128),
                                                    batch_size=64,
                                                    class_mode='categorical')
# Reading the testing set
test_set = test_datagen.flow_from_directory('dataset/gender/test',
                                            target_size=(128, 128),
                                            batch_size=64,
                                            class_mode='categorical')

############################
# STEP 4: Training The Model
############################

# Finally training the model
# For better accuracy adjust the epochs
model.fit_generator(train_generator,
                    steps_per_epoch=2800,
                    epochs=5,
                    validation_data=test_set,
                    validation_steps=800)

##########################
# STEP 5: Saving The Model
##########################

# Saving the model
model.save('weights/gender_mobile.h5')
