
import time
import logging
import spacy
import numpy as np
from creme import stream
from sklearn.neural_network import MLPClassifier

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
    alpha=1e-4, hidden_layer_sizes=(64, 32, 16,), random_state=1, batch_size=1, verbose= False,
    max_iter=1, warm_start=False)

predictions = []
acc = 0.0
crr = 0.0
count = 0.0
data_x = []
data_y = []
classes = [0, 1, 2, 3, 4]

train_start = time.time()

for (i, (X, y)) in enumerate(dataset):

    start_time = time.time()
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
    start_time = time.time()
    logging.info('\tTraining on data instance')
    classifier.partial_fit(X_embedding, y_encodding, classes)
    stop_time = time.time()
    elapsed_time = stop_time - start_time
    logging.info('\tFinished in {0} seconds.'.format(elapsed_time))
    logging.info('\tPredicting...')
    prediction = classifier.predict(X_embedding)
    predictions.append(prediction)
    stop_time = time.time()
    elapsed_time = stop_time - start_time
    logging.info('\tFinished in {0} seconds.'.format(elapsed_time))

    count += 1
    if prediction == y_encodding[0]:
        crr +=1

    acc = crr / count

    logging.warn('\tAccuracy after training on {0} data instances: {1}'.format(i, acc))

train_end = time.time()
train_time = train_end - train_start
logging.warn('\tTraining time: {0}'.format(train_time))
