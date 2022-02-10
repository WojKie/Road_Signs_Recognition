import pandas as pd
import numpy as np
import tensorflow as tf
import os
import cv2
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn import metrics
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, MaxPool2D, Flatten, Dense, Dropout

#dataframe treningowy
train_path_csv = 'D:\Wojtek\Traffic_signs\Train.csv'
#datafrmae testowy
test_path_csv = 'D:\Wojtek\Traffic_signs\Test.csv'

train_path_folder = 'D:\Wojtek\Traffic_signs\Train'
test_path_folder = 'D:\Wojtek\Traffic_signs\Test'

train_df = pd.read_csv(train_path_csv, sep = ',')
test_df = pd.read_csv(test_path_csv, sep = ',')

print(train_df.head(5), '\n', test_df.head(5))

train_images = []
train_labels = []
classes = 43
SIZE = 32

for i in os.listdir(train_path_folder):
    dir = train_path_folder + '\\' + i
    for j in os.listdir(dir):
        image_path = dir + '\\' + j
        image = cv2.imread(image_path, -1)
        img_x = cv2.resize(image, (SIZE, SIZE), interpolation = cv2.INTER_NEAREST)
        train_images.append(img_x)
        train_labels.append(i)

test_images = []
test_labels = []

for i in os.listdir(test_path_folder):
    try:
        image_path = test_path_folder + '\\' + i
        image = cv2.imread(image_path, -1)
        img_x = cv2.resize(image, (SIZE, SIZE), interpolation = cv2.INTER_NEAREST)
        test_images.append(img_x)
    except Exception as e:
        print(str(e))



plot4 = plt
plot4.figure(figsize = (7, 7))
plot4.subplot(221), plot4.imshow(train_images[13])
plot4.subplot(222), plot4.imshow(train_images[69])
plot4.subplot(223), plot4.imshow(train_images[420])
plot4.subplot(224), plot4.imshow(train_images[1337])

#plot4.savefig('git_'+str(SIZE)+'.png')


train_images = np.array(train_images)
train_labels = np.array(train_labels)

test_images = np.array(test_images)
test_labels = np.array(test_df['ClassId'].values)

print(test_images.shape, test_labels.shape)
print(train_images.shape, train_labels.shape)

train_img, test_img, train_lbl, test_lbl = train_test_split(train_images, train_labels, test_size = 0.3, random_state = 11)

print((train_img.shape, train_lbl.shape), (test_img.shape, test_lbl.shape))

train_lbl_copy = train_lbl.copy()
test_lbl_copy = test_lbl.copy()

#konwersja labeli poprzez one hot encoding
train_lbl = to_categorical(train_lbl, 43)
test_lbl = to_categorical(test_lbl, 43)

#model
model_cnn = Sequential()
model_cnn.add(Conv2D(filters = 32, kernel_size = (5, 5), activation = 'relu', input_shape = (SIZE, SIZE, 3), padding = 'same'))
model_cnn.add(Conv2D(filters = 32, kernel_size = (5, 5), activation = 'relu', padding = 'same'))
model_cnn.add(MaxPool2D(pool_size = (2, 2)))
model_cnn.add(Dropout(rate = 0.3))

model_cnn.add(Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu', padding = 'same'))
model_cnn.add(Conv2D(filters = 64, kernel_size = (3, 3), activation = 'relu', padding = 'same'))
model_cnn.add(MaxPool2D(pool_size = (2, 2)))
model_cnn.add(Dropout(rate = 0.3))

model_cnn.add(Flatten()) #encoder do tego
model_cnn.add(Dense(256, activation = 'relu'))
model_cnn.add(Dropout(rate = 0.3))
model_cnn.add(Dense(43, activation = 'softmax'))

model_cnn.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

model_cnn.summary()

history_cnn = model_cnn.fit(train_img, train_lbl, batch_size = 64, epochs = 15, validation_data = (test_img, test_lbl))


print(type(test_lbl_copy[250]))
tlc_int = test_lbl_copy.astype('int32')
print(type(tlc_int[250]))

cnn_pred = model_cnn.predict_classes(test_img)


cnn_recall = metrics.recall_score(tlc_int, cnn_pred, average = 'macro')

cnn_precision = metrics.precision_score(tlc_int, cnn_pred, average = 'macro')

cnn_f1 = metrics.f1_score(tlc_int, cnn_pred, average = 'macro')

cnn_acc = metrics.accuracy_score(tlc_int, cnn_pred)


print('CNN Recall: ', cnn_recall, '\n', 'CNN Precision: ', cnn_precision, '\n', 'CNN F1: ', cnn_f1, '\n', 'CNN Accuracy: ', cnn_acc)

metryki = ['Recall', 'Precision', 'F1', 'Accuracy']

metryki_cnn = [cnn_recall, cnn_precision, cnn_f1, cnn_acc]

#dodać jak rosły wszystkie parametry dla każdego powtórzenia

def extract_layers(main_model, starting_layer_index, ending_layer_index):
    new_model = Sequential()
    for indexx in range(starting_layer_index, ending_layer_index + 1):
        current_layer = main_model.get_layer(index = indexx)
        new_model.add(current_layer)
    return new_model

my_encoder = extract_layers(model_cnn, 0, 8)
my_encoder.summary()

print('train_img shape: ', train_img.shape)
print('test_img shape: ', test_img.shape)
print('test_images shape: ', test_images.shape)


my_encoder.compile(loss = 'categorical_crossentropy', optimizer = 'adam', metrics = ['accuracy'])

#TU BĘDZIE ENCODOWANIE ORYGINALNYCH OBRAZÓW
X = my_encoder.predict(train_images)

#Y = train_labels.copy()
Y = np.array(train_df['ClassId'].values)

X_tr, X_ts, Y_tr, Y_ts = train_test_split(X, Y, test_size = 0.3, random_state = 42)


X_test = my_encoder.predict(test_images)
Y_test = test_labels.copy()

print(X.shape, X_test.shape)


##########################################


import xgboost as xgb
from sklearn import preprocessing, svm

xgb_cl = xgb.XGBClassifier()


xgb_cl.fit(X_tr, Y_tr)

xgb_cl.score(X_ts, Y_ts)


xgb_pred = xgb_cl.predict(X_ts)


xgb_recall = metrics.recall_score(Y_ts, xgb_pred, average = 'macro')

xgb_precision = metrics.precision_score(Y_ts, xgb_pred, average = 'macro')

xgb_f1 = metrics.f1_score(Y_ts, xgb_pred, average = 'macro')

xgb_acc = metrics.accuracy_score(Y_ts, xgb_pred)

print('XGB Recall: ', xgb_recall, '\n', 'XGB Precision: ', xgb_precision, '\n', 'XGB F1: ', xgb_f1, '\n', 'XGB Accuracy: ', xgb_acc)

metryki_xgb = [xgb_recall, xgb_precision, xgb_f1, xgb_acc]

################################################

from sklearn.ensemble import RandomForestClassifier

random_las = RandomForestClassifier(n_estimators = 350, random_state = 42, criterion = 'entropy')


print(Y.shape, X.shape)


#random_las.fit(img_enc, train_labels)
random_las.fit(X_tr, Y_tr)


#random_las.predict(test_images_enc)
random_las.predict(X_ts)

accuracy_rf = metrics.accuracy_score(Y_ts, random_las.predict(X_ts))

print(accuracy_rf)

rf_pred = random_las.predict(X_ts)

rf_recall = metrics.recall_score(Y_ts, rf_pred, average = 'macro')

rf_precision = metrics.precision_score(Y_ts, rf_pred, average = 'macro')

rf_f1 = metrics.f1_score(Y_ts, rf_pred, average = 'macro')

rf_acc = metrics.accuracy_score(Y_ts, rf_pred)

print('RF Recall: ', rf_recall, '\n', 'RF Precision: ', rf_precision, '\n', 'RF F1: ', rf_f1, '\n', 'RF Accuracy: ', rf_acc)


metryki_rf = [rf_recall, rf_precision, rf_f1, rf_acc]

################################################

svm_model = svm.NuSVC(nu = 0.05, kernel = 'rbf', gamma = 0.00001, random_state = 42)

svm_model.fit(X_tr, Y_tr)

svm_pred = svm_model.predict(X_ts)

svm_recall = metrics.recall_score(Y_ts, svm_pred, average = 'macro')

svm_precision = metrics.precision_score(Y_ts, svm_pred, average = 'macro')

svm_f1 = metrics.f1_score(Y_ts, svm_pred, average = 'macro')

svm_acc = metrics.accuracy_score(Y_ts, svm_pred)


print('SVM Recall: ', svm_recall, '\n', 'SVM Precision: ', svm_precision, '\n', 'SVM F1: ', svm_f1, '\n', 'SVM Accuracy: ', svm_acc)


metryki_svm = [svm_recall, svm_precision, svm_f1, svm_acc]


wykres_cnn = plt.figure()
ax = wykres_cnn.add_axes([0,0,1.5,1.5])
ax.bar(metryki, metryki_cnn, color = 'gray')
ax.set_xlabel('Metryki', fontsize='large')
ax.set_ylabel('Dokładność', fontsize='large')
ax.set_title('Wykres metryk CNN ' + str(SIZE), fontsize='large', pad=20)
for i in ax.patches:
    ax.text(i.get_x(), i.get_height()+.03,str(round((i.get_height()), 5)), fontsize=10,color='black')
plt.show()

wykres_xgb = plt.figure()
bx = wykres_xgb.add_axes([0,0,1.5,1.5])
bx.bar(metryki, metryki_xgb, color = 'purple')
bx.set_xlabel('Metryki', fontsize='large')
bx.set_ylabel('Dokładność', fontsize='large')
bx.set_title('Wykres metryk XGB ' + str(SIZE), fontsize='large', pad=20)
for i in bx.patches:
    bx.text(i.get_x(), i.get_height()+.03,str(round((i.get_height()), 5)), fontsize=10,color='black')
plt.show()


wykres_rf = plt.figure()
cx = wykres_rf.add_axes([0,0,1.5,1.5])
cx.bar(metryki, metryki_rf, color = 'orange')
cx.set_xlabel('Metryki', fontsize='large')
cx.set_ylabel('Dokładność', fontsize='large')
cx.set_title('Wykres metryk RF ' + str(SIZE), fontsize='large', pad=20)
for i in cx.patches:
    cx.text(i.get_x(), i.get_height()+.03,str(round((i.get_height()), 5)), fontsize=10,color='black')
plt.show()


wykres_svm = plt.figure()
dx = wykres_svm.add_axes([0,0,1.5,1.5])
dx.bar(metryki, metryki_svm, color = 'plum')
dx.set_xlabel('Metryki', fontsize='large')
dx.set_ylabel('Dokładność', fontsize='large')
dx.set_title('Wykres metryk SVM ' + str(SIZE), fontsize='large', pad=20)
for i in dx.patches:
    dx.text(i.get_x(), i.get_height()+.03,str(round((i.get_height()), 5)), fontsize=10,color='black')
plt.show()
