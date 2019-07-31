
import time
import logging
import spacy
import numpy as np
from creme import stream
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

PATH_TO_CSV = 'bbc-text.csv'

start_time = time.time()
logger = logging.getLogger()
logger.setLevel(logging.WARN)
logging.warn('\tLoading word embeddings and data streamer...')
nlp = spacy.load('en_core_web_md')
encodings = {'tech':0, 'business':1, 'sport':2, 'entertainment':3, 'politics':4}
types = {"category":str}
dataset = stream.iter_csv(PATH_TO_CSV, target_name="category", types=types)
stop_time = time.time()
elapsed_time = stop_time - start_time
logging.info('\tFinished in {0} seconds.'.format(elapsed_time))

classifier = MLPClassifier(activation='tanh', learning_rate='constant',
    alpha=1e-4, hidden_layer_sizes=(15,), random_state=1, batch_size=16, verbose= False,
    max_iter=20, warm_start=True)

predictions = []
acc = 0.0
crr = 0.0
count = 0.0
data_x = []
data_y = []
classes = [0, 1, 2, 3, 4]

train_start = time.time()

for (i, (X, y)) in enumerate(dataset):

    logging.info('\tGetting data instance embeddings using creme streamer and GloVe model.')
    data_instance = []
    X_embedding = np.array([nlp(X['text']).vector])
    y_encodding = np.array([encodings[y]]).ravel()
    data_instance.append(X_embedding)
    data_instance.append(y_encodding)
    data_x.append(np.asarray(X_embedding))
    data_y.append(np.asarray(y_encodding))
    stop_time = time.time()
    elapsed_time = stop_time - start_time
    logging.info('\tFinished in {0} seconds.'.format(elapsed_time))
    if i%200 == 0 :
        logging.warn('\tData Number: {0}'.format(i))

x_data = np.asarray(data_x).reshape(-1, 300)
y_data = np.asarray(data_y).ravel()
x_train, x_test, y_train, y_test = train_test_split(x_data, y_data, test_size= 0.25, random_state=27)
print(x_data.shape)
print(y_data.shape)
print(x_train.shape)
print(y_train.shape)
classifier.fit(x_train, y_train)
predictions = classifier.predict(x_test)
acc = accuracy_score(y_test, predictions)
train_end = time.time()
train_time = train_end - train_start
logging.warn('\tTraining time: {0}'.format(train_time))
logging.warn('\tAccuracy: {0}'.format(acc))
