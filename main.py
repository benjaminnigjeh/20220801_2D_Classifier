import pandas as pd
import numpy as np
import tensorflow as TF
import keras
from keras import utils
import datetime
from keras.callbacks import TensorBoard


# set-up the tensorboard with a unique name to monitor the model performance
NAME = "AI-Second"
tensorboard = TensorBoard(log_dir='C:/AIML/logs/{}'.format(NAME))


# CSV file has to be normazlied beforehand proteins are at columns and samples ar at rows
#import data as CSV file and split it on training and evaluation dataframes

file_path = 'C:/AIML/abcd.csv'
df = pd.read_csv(file_path)
X_ev = df.sample(frac=0.2, random_state=1337)
X_tr = df.drop(X_ev.index)
X = X_tr.copy()
Y = X.pop("target")
X_ev = X_ev.copy()
Y_ev = X_ev.pop("target")


#convert dataframes to numpy arrays
X_tf = np.array(X)
Y_tf = np.array(Y)
X_ev = np.array(X_ev)
Y_ev = np.array(Y_ev)

#sequential dense nn model
model = TF.keras.models.Sequential()
model.add(TF.keras.layers.Dense(128, activation=TF.nn.relu))
model.add(TF.keras.layers.Dense(1280, activation=TF.nn.relu))
model.add(TF.keras.layers.Dense(4, activation=TF.nn.softmax))


#compile and fit the model
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics='accuracy')

model.fit(X_tf, Y_tf, epochs=40, callbacks=[tensorboard])


#evaluate the model
val_loss, val_acc = model.evaluate(X_ev, Y_ev)
print(val_acc)
print(val_loss)

#save and reload the model
model.save('C:/AIML/AI_classifier')
new_model = TF.keras.models.load_model('C:/AIML/AI_classifier')

#predict the probbalities
Predictions = model.predict(X_tf)
print(np.argmax(Predictions[50]))




TF.keras.utils.plot_model(
    new_model,
    to_file="model.png",
    show_shapes=False,
    show_dtype=False,
    show_layer_names=True,
    rankdir="TB",
    expand_nested=False,
    dpi=96,
    layer_range=None,
    show_layer_activations=False,
)

new_model.summary()

