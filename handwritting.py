import sys
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import cohen_kappa_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

df = pd.read_csv('A_Z Handwritten Data.csv').astype('float32')

# x and y split
y = df.iloc[:, 0]
x = df.iloc[:, 1:]

# train test split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 420)

# reshaping from 784 array to 2d matrix of 28 x 28 x 1
x_train = np.reshape(x_train.values, (x_train.shape[0], 28,28))
x_train = x_train.reshape(x_train.shape[0],x_train.shape[1],x_train.shape[2],1)
x_test = np.reshape(x_test.values, (x_test.shape[0], 28,28))
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2],1)


x_train, x_test = x_train / 255.0, x_test / 255.0 # scaling
y_train = tf.keras.utils.to_categorical(y_train)
y_test = tf.keras.utils.to_categorical(y_test)

print(x_train.shape)
print(x_test.shape)
print(y_train.shape)
print(y_test.shape)

# Create a convolutional neural network




model = tf.keras.models.Sequential([

    # Convolutional layer. Learn 32 filters using a 3x3 kernel
    tf.keras.layers.Conv2D(
        32, (3, 3), activation="relu", input_shape=(28,28,1)
    ),
    # Max-pooling layer, using 2x2 pool size
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    # Convolutional layer. Learn 64 filters using a 3x3 kernel
    tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding = 'same'),
    # Max-pooling layer, using 2x2 pool size
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding = 'valid'),
    # Max-pooling layer, using 2x2 pool size
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2)),

    # Flatten units
    tf.keras.layers.Flatten(),

    # Add an output layer with output units for all 10 digits
    tf.keras.layers.Dense(64, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(128, activation="relu"),
    tf.keras.layers.Dropout(0.5),
    tf.keras.layers.Dense(26, activation="softmax"), # binary output: sigmoid, # 26 output: not sigmoid, i.e. softmax
])


# Train neural network
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)


history = model.fit(x_train, y_train, epochs=5)


# Evaluate neural network performance
model.evaluate(x_test,  y_test, verbose=2)

'''not my code: this if for evaluation which is not included in keras'''


# predict probabilities for test set
yhat_probs = model.predict(x_test, verbose=0)
# predict crisp classes for test set
predict_x=model.predict(x_test) 
yhat_classes=np.argmax(predict_x,axis=1)



# reduce to 1d array
yhat_probs = yhat_probs[:, 0]
#yhat_classes = yhat_classes[:, 0]

print('\n\n',yhat_classes)
print('\n\n')
# accuracy: (tp + tn) / (p + n)


y_test=np.argmax(y_test, axis=1)


accuracy = accuracy_score(y_test, yhat_classes)

precision = precision_score(y_test, yhat_classes, average='macro')

recall = recall_score(y_test, yhat_classes, average='macro')

f1 = f1_score(y_test, yhat_classes, average='macro')


print('Accuracy: %f' % accuracy)
print('Precision: %f' % precision)
print('Recall: %f' % recall)
print('F1 score: %f' % f1)


print(classification_report(y_test, yhat_classes))


plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'val'], loc='upper left')
plt.show()

# Save model to file
if len(sys.argv) == 2:
    filename = sys.argv[1]
    model.save(filename)
    print(f"Model saved to {filename}.")



