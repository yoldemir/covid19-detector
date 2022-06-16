from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPool2D, Dropout, Flatten, Dense, Rescaling, RandomFlip, RandomRotation, RandomZoom
from tensorflow.keras.utils import image_dataset_from_directory

model = Sequential()
model.add(Rescaling(scale=1./255))
model.add(RandomRotation(0.1))
model.add(RandomZoom(0.1))
model.add(Conv2D(32, kernel_size=(3, 3), activation="relu"))
model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Conv2D(128, kernel_size=(3, 3), activation="relu"))
model.add(MaxPool2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(1, activation="sigmoid"))

model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
model.build((32, 224, 224, 3))
model.summary()

train_ds = image_dataset_from_directory('CovidDataset/Train', seed=42, image_size=(224, 224), batch_size=32)
val_ds = image_dataset_from_directory('CovidDataset/Val', seed=42, image_size=(224, 224), batch_size=32)

model.fit(train_ds, epochs=10, validation_data=val_ds, validation_steps=2)



