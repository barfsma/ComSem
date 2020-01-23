import pandas as pd
from keras.models import Sequential
from keras.layers.core import Dense, Activation
from keras.optimizers import SGD
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

with open("SICK/SICK_train.txt", "rb") as f:
    train_df = pd.read_csv(f, sep = "\t").drop(columns = ["pair_ID"])

with open("SICK/SICK_trial.txt", "rb") as f:
    dev_df = pd.read_csv(f, sep = "\t").drop(columns = ["pair_ID"])


train_df["text"] = train_df["sentence_A"] + train_df["sentence_B"]
dev_df["text"] = dev_df["sentence_A"] + dev_df["sentence_B"]

X, Y = train_df[["text", "entailment_judgment"]], dev_df[["text", "entailment_judgment"]]

X_train, X_test, Y_train, Y_test = train_df["text"], dev_df["text"], train_df["entailment_judgment"], dev_df["entailment_judgment"]

"""
pipeline = Pipeline([("vec", TfidfVectorizer()),
                     ("clf", RandomForestClassifier())])

pipeline.fit(X_train, Y_train)
Y_pred = pipeline.predict(X_test)
"""

model = Sequential()
model.add(Dense(input_dim=X.shape[1], units=Y.shape[1]))
model.add(Activation("linear"))
sgd = SGD(lr=0.01)
loss_function = "mean_squared_error"
model.compile(loss=loss_function, optimizer=sgd, metrics=["accuracy"])
# Train the perceptron
model.fit(X_train, Y_train, verbose=1, epochs=1, batch_size=32)
# Get predictions
Y_pred = model.predict(Xtest)
# Convert to numerical labels to get scores with sklearn in 6-way setting
Y_pred = numpy.argmax(Y_pred, axis=1)
Y_test = numpy.argmax(Y_test, axis=1)

print("Macro F1-score:\t {}".format(f1_score(Y_test, Y_pred, average = "macro")))
print("Macro precision: {}".format(precision_score(Y_test, Y_pred, average = "macro")))
print("Macro recall:\t {}\n".format(recall_score(Y_test, Y_pred, average = "macro")))
print(classification_report(Y_test, Y_pred))