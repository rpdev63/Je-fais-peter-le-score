# -*- coding: utf-8 -*-
"""
Example script

Script to perform some corrections in the brief audio project

Created on Fri Jan 27 09:08:40 2023

@author: ValBaron10
"""

# Import
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from functions.features import compute_features
import time
import seaborn as sns
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import confusion_matrix
import pandas as pd



def log_normalization(data):
    return np.log(data + 1)

# Set the paths to the files 
data_path = "Data/"

# Names of the classes
classes_paths = ["Cars/", "Trucks/"]
classes_names = ["car", "truck"]
cars_list = [4,5,7,9,10,15,20,21,23,26,30,38,39,44,46,48,51,52,53,57]
trucks_list = [2,4,10,11,13,20,22,25,27,30,31,32,33,35,36,39,40,45,47,48]
nbr_of_sigs = 20 # Nbr of sigs in each class
seq_length = 0.2 # Nbr of second of signal for one sequence
nbr_of_obs = int(nbr_of_sigs*10/seq_length) # Each signal is 10 s long

# Go to search for the files
learning_labels = []
for i in range(2*nbr_of_sigs):
    if i < nbr_of_sigs:
        name = f"{classes_names[0]}{cars_list[i]}.wav"
        class_path = classes_paths[0]
    else:
        name = f"{classes_names[1]}{trucks_list[i - nbr_of_sigs]}.wav"
        class_path = classes_paths[1]

    # Read the data and scale them between -1 and 1
    fs, data = sio.wavfile.read(data_path + class_path + name)
    data = data.astype(float)
    data = data/32768

    # Cut the data into sequences (we take off the last bits)
    data_length = data.shape[0]
    nbr_blocks = int((data_length/fs)/seq_length)
    seqs = data[:int(nbr_blocks*seq_length*fs)].reshape((nbr_blocks, int(seq_length*fs)))

    for k_seq, seq in enumerate(seqs):
        # Compute the signal in three domains
        sig_sq = seq**2
        sig_t = seq / np.sqrt(sig_sq.sum())
        sig_f = np.absolute(np.fft.fft(sig_t))
        sig_c = np.absolute(np.fft.fft(sig_f))

        # Compute the features and store them
        features_list = []
        N_feat, features_list = compute_features(sig_t, sig_f[:sig_t.shape[0]//2], sig_c[:sig_t.shape[0]//2], fs)
        features_vector = np.array(features_list)[np.newaxis,:]

        if k_seq == 0 and i == 0:
            learning_features = features_vector
            learning_labels.append(classes_names[0])
        elif i < nbr_of_sigs:
            learning_features = np.vstack((learning_features, features_vector))
            learning_labels.append(classes_names[0])
        else:
            learning_features = np.vstack((learning_features, features_vector))
            learning_labels.append(classes_names[1])

#création d'un dataframe et d'un csv
df = pd.DataFrame(learning_features, columns=[f"feature{i +1}" for i in range(learning_features.shape[1])])
df['label'] = learning_labels

# df.to_pickle(r"Data/data")

# Seperate features and label
X = df.select_dtypes(include=['int', 'float'])
y = df["label"]

# Logarithmic normalization
X_log = log_normalization(X)
index_before_drop = X_log.index
X_log.dropna(inplace=True)
index_after_drop = X_log.index
dropped_index = set(index_before_drop) - set(index_after_drop)
y_log = y[~y.index.isin(dropped_index)]

# Separate data in train and test
X_train, X_test, y_train, y_test = train_test_split(X_log, y_log, test_size = 0.2, stratify=y_log, random_state=42)



# Initialize the model
model = ExtraTreesClassifier(bootstrap= False, criterion='entropy', max_depth= None, max_features= None, min_samples_split= 3, n_estimators= 300)

# Perform 5-fold cross validation and get the accuracy scores for each fold
ts = time.time()
model.fit(X_train, y_train)    
te = time.time()
scores = cross_val_score(model, X_train, y_train, cv=5)


# Print score
print(f"Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
print("Training time : {}s".format(round(te - ts,2),'s'))

y_pred = model.predict(X_test)

# Plot the confusion matrix using seaborn
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix')
plt.show()
