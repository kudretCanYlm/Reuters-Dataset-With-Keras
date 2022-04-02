# The Reuters dataset
import copy
from keras.datasets import reuters
from keras.utils.np_utils import to_categorical
from keras import models, layers, optimizers, losses
import matplotlib.pyplot as plt
import numpy as np

(train_data, train_labels), (test_data,
                             test_labels) = reuters.load_data(num_words=10000)

print("length train data: ", len(train_data))
print("length test data:", len(test_data))
print(test_data[15])


def write_text(data, number):
    word_index = reuters.get_word_index()
    reserve_word_index = dict([(value, key)
                              for (key, value) in word_index.items()])
    decoded_text = " ".join([reserve_word_index.get(i-3, "?")
                             for i in data[number]])
    print("text: ", decoded_text)


write_text(test_data, 15)

# Preparing the data


def vectorize_sequences(sequences, dimension=10000):
    results = np.zeros((len(sequences), dimension))
    for i, sequence in enumerate(sequences):
        results[i, sequence] = 1.
    return results


def to_one_hot(labels, dimension=46):
    results = np.zeros((len(labels), dimension))
    for i, label in enumerate(labels):
        results[i, label] = 1.
    return results


x_train = vectorize_sequences(train_data)
x_test = vectorize_sequences(test_data)


one_hot_train_labels = to_one_hot(train_labels)
one_hot_test_labels = to_one_hot(test_labels)

print("size of x_train: ", x_train.shape)
print("size of x_test: ", x_test.shape)
print("size of one_hot_train_labels: ", one_hot_train_labels.shape)
print("size of one_hot_test_labels: ", one_hot_test_labels.shape)

# yukardaki to_categorical ile de yapılır
# one_hot_train_labels = to_categorical(train_labels)
# one_hot_test_labels = to_categorical(test_labels)

# Building your network

model = models.Sequential()
model.add(layers.Dense(64, activation="relu", input_shape=(10000,)))
model.add(layers.Dropout(0.5))
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dropout(0.6))
model.add(layers.Dense(46, activation="softmax"))

model.compile(optimizer=optimizers.adam_v2.Adam(learning_rate=0.001),
              loss=losses.categorical_crossentropy, metrics=["acc"])

# Validating your approach
x_val = x_train[:1000]
partial_x_train = x_train[1000:]

y_val = one_hot_train_labels[:1000]
partial_y_train = one_hot_train_labels[1000:]

history = model.fit(partial_x_train, partial_y_train,
                    batch_size=512, epochs=20, validation_data=(x_val, y_val))

results = model.evaluate(x_test, one_hot_test_labels)
print('result: ', results)
# Plotting the training and validation loss
history_dict = history.history
print(history_dict.keys())
loss_values = history.history["loss"]
val_loss_values = history.history["val_loss"]
acc = history.history["acc"]

epochs = range(1, len(acc)+1)

plt.plot(epochs, loss_values, "bo", label="Training loss")

plt.plot(epochs, val_loss_values, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.show()

# Plotting the training and validation accuracy
plt.clf()  # <------------ lears the figure
acc_values = history.history["acc"]
val_acc_values = history.history["val_acc"]

plt.plot(epochs, acc_values, "bo", label="Training acc")
plt.plot(epochs, val_acc_values, "b", label="Validation acc")
plt.title("Training and validation accuracy")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.show()

# so the results seem pretty good, at least when compared to a random baseline:
test_labels_copy = copy.copy(test_labels)
np.random.shuffle(test_labels_copy)
hits_array = np.array(test_labels) == np.array(test_labels_copy)
print(float(np.sum(hits_array))/len(test_labels))

# Generating predictions on new data
predictions = model.predict(x_test)

# Each entry in predictions is a vector of length 46:
print(predictions[0].shape)

#The coefficients in this vector sum to 1:
print(np.sum(predictions[0]))

#The largest entry is the predicted class—the class with the highest probability:
print(np.argmax(predictions[0]))