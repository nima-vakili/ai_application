from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing import image

mnist = tf.keras.datasets.mnist
(X_train, y_train), (X_test, y_test) = mnist.load_data()


X_train = X_train.reshape(X_train.shape[0], X_train.shape[1],
                          X_train.shape[2], 1)
X_train = X_train / 255.0
X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], X_test.shape[2], 1)
X_test = X_test / 255.0

X_train = image.smart_resize(X_train, (64, 64))
X_test = image.smart_resize(X_test, (64, 64))

encoder = LabelEncoder().fit(y_train)
y_train_cat = encoder.transform(y_train)
y_test_cat = encoder.transform(y_test)

y_train_oh = to_categorical(y_train_cat)
y_test_oh = to_categorical(y_test_cat)

model = load_model('../ai_library/model_save.h5')
model.pop()
model.add(layers.Dense(10, activation='softmax'))
model.summary()

for i in range(len(model.layers) - 5):
    model.layers[i].trainable = False


def compile_model(model):
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics="accuracy")
    return model


model = compile_model(model)


es = EarlyStopping(patience=5, monitor='val_accuracy',
                   restore_best_weights=True)

history = model.fit(X_train, y_train_oh,
                    batch_size=10,
                    epochs=10,
                    validation_split=0.3,
                    callbacks=[es])

model.evaluate(X_test, y_test_oh, verbose=0)
predictions = model.predict(X_test)

model.save('model_save_fromTrained.h5')
