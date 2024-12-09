
import os

# Only the TensorFlow backend supports string inputs.
os.environ["KERAS_BACKEND"] = "tensorflow"

import pathlib
import numpy as np
import tensorflow.data as tf_data
import keras
from keras import layers
from keras.layers import Embedding
import matplotlib.pyplot as plt

def create_parent_directory():
    data_path = keras.utils.get_file(
        "news20.tar.gz",
        "http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/news20.tar.gz",
        untar=True,
    )
    data_dir = pathlib.Path(data_path).parent / "20_newsgroup"
    return data_dir

def data2list(data_dir):
    samples = []
    labels = []
    class_names = []
    class_index = 0
    for dirname in sorted(os.listdir(data_dir)):
        class_names.append(dirname)
        dirpath = data_dir / dirname
        fnames = os.listdir(dirpath)
        for fname in fnames:
            fpath = dirpath / fname
            f = open(fpath, encoding="latin-1")
            content = f.read()
            lines = content.split("\n")
            lines = lines[10:]
            content = "\n".join(lines)
            samples.append(content)
            labels.append(class_index)
        class_index += 1
    return samples, labels, class_names

def shuffle_data(samples, labels):
    '''samples = samples[:int(len(samples)/3)]
    labels = labels[:int(len(labels)/3)]'''
    seed = 1337
    rng = np.random.RandomState(seed)
    rng.shuffle(samples)
    rng = np.random.RandomState(seed)
    rng.shuffle(labels)
    return samples, labels

def get_trainNval_data(samples, labels, validation_split=0.2):
    num_validation_samples = int(validation_split * len(samples))
    train_samples = samples[:-num_validation_samples]
    val_samples = samples[-num_validation_samples:]
    train_labels = labels[:-num_validation_samples]
    val_labels = labels[-num_validation_samples:]
    return train_samples, val_samples, train_labels, val_labels

def vectorize_data(train_samples, val_samples):
    vectorizer = layers.TextVectorization(max_tokens=20000, output_sequence_length=200)
    text_ds = tf_data.Dataset.from_tensor_slices(train_samples).batch(128)
    vectorizer.adapt(text_ds)
    voc = vectorizer.get_vocabulary()
    word_index = dict(zip(voc, range(len(voc))))
    path_to_glove_file = "/glove.6B.100d.txt"
    embeddings_index = {}
    with open(path_to_glove_file) as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, "f", sep=" ")
            a = coefs
            embeddings_index[word] = coefs
    return vectorizer, voc, word_index, embeddings_index

def create_embedding_matrix(word_index, embeddings_index, num_tokens, embedding_dim):
    hits = 0
    misses = 0
    # Prepare embedding matrix
    embedding_matrix = np.zeros((num_tokens, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros.
            # This includes the representation for "padding" and "OOV"
            embedding_matrix[i] = embedding_vector
            hits += 1
        else:
            misses += 1
    return embedding_matrix

def create_model(int_sequences_input, preds):
    model = keras.Model(int_sequences_input, preds)
    #model.summary()
    return model

def train_model(vectorizer, train_samples, train_labels, val_samples, val_labels, model, epochs):
    x_train = vectorizer(np.array([[s] for s in train_samples])).numpy()
    x_val = vectorizer(np.array([[s] for s in val_samples])).numpy()

    y_train = np.array(train_labels)
    y_val = np.array(val_labels)
    model.compile(
        loss="sparse_categorical_crossentropy", optimizer="rmsprop", metrics=["acc"]
    )
    history = model.fit(x_train, y_train, batch_size=128, epochs=epochs, validation_data=(x_val, y_val))
    return history.history['loss'], history.history['acc'], history.history['val_loss'], history.history['val_acc']

# create dumb word embedding
def create_one_hot_embeddings_matrix(word_index, num_tokens):
    one_hot_embeddings_index = {}
    for word, i in word_index.items():
        one_hot_vector = np.zeros(num_tokens)
        one_hot_vector[i] = 1.0
        one_hot_embeddings_index[word] = one_hot_vector
    embedding_matrix = np.zeros((num_tokens, num_tokens))
    for word, i in word_index.items():
        embedding_vector = one_hot_embeddings_index.get(word)
        if embedding_vector is not None:
            embedding_matrix[i] = embedding_vector
    vertical_sum = np.sum(embedding_matrix, axis=0)
    print("Vertical sum of embedding matrix:", vertical_sum)
    horizontal_sum = np.sum(embedding_matrix, axis=1)
    print("Horizontal sum of embedding matrix:", horizontal_sum)
    return embedding_matrix

def create_embedding_layer(class_names, embedding_matrix, num_tokens, embedding_dim, dumb):
    if dumb:
        embedding_matrix = np.eye(num_tokens-2)
        embedding_dim = num_tokens-2
        num_tokens = num_tokens-2
    embedding_layer = Embedding(
        num_tokens,
        embedding_dim,
        trainable=True,
    )
    embedding_layer.build((1,))
    embedding_layer.set_weights([embedding_matrix])
    int_sequences_input = keras.Input(shape=(None,), dtype="int32")
    embedded_sequences = embedding_layer(int_sequences_input)
    x = layers.Conv1D(128, 5, activation="relu")(embedded_sequences)
    x = layers.MaxPooling1D(5)(x)
    x = layers.Conv1D(128, 5, activation="relu")(x)
    x = layers.MaxPooling1D(5)(x)
    x = layers.Conv1D(128, 5, activation="relu")(x)
    x = layers.GlobalMaxPooling1D()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    preds = layers.Dense(len(class_names), activation="softmax")(x)
    return int_sequences_input, preds

def plot(train, val, value_type):
    train_range = range(len(train))
    val_range = range(len(val))
    plt.plot(train_range, train, color = 'r', label='Training ' + value_type)
    plt.plot(val_range, val, color = 'b', label='Validation ' + value_type)
    plt.title('Training and Validation ' + value_type)
    plt.legend(loc='upper left', fontsize='small')
    plt.xlabel('Epochs')
    plt.ylabel(value_type)
    plt.show()

def plot_dif_method(train, val, dumb_train, dumb_val):
    train_range = range(len(train))
    val_range = range(len(val))
    plt.plot(train_range, train, color = 'r', label='Dumb training loss')
    plt.plot(val_range, val, color = 'r', linestyle='--', label='Dumb validation loss')
    plt.plot(train_range, dumb_train, color = 'b', label='GloVE training loss')
    plt.plot(val_range, dumb_val, color = 'b', linestyle='--', label='GloVE validation loss')
    plt.title('Training and Validation loss')
    plt.legend(loc='upper left', fontsize='small')
    plt.xlabel('Epochs')
    plt.ylabel('loss')
    plt.show()

def main():
    data_dir = create_parent_directory()
    epochs = 20
    samples, labels, class_names = data2list(data_dir)
    samples, labels = shuffle_data(samples, labels)
    train_samples, val_samples, train_labels, val_labels = get_trainNval_data(samples, labels)
    vectorizer, voc, word_index, embeddings_index = vectorize_data(train_samples, val_samples)

    num_tokens = len(voc) + 2
    embedding_dim = 100

    embedding_matrix = create_embedding_matrix(word_index, embeddings_index, num_tokens, embedding_dim)
    #dumb_embedding_matrix = create_one_hot_embeddings_matrix(word_index, num_tokens)

    int_sequences_input, preds = create_embedding_layer(class_names, embedding_matrix, num_tokens, embedding_dim, False)
    model = create_model(int_sequences_input, preds)
    loss, acc, val_loss, val_acc = train_model(vectorizer, train_samples, train_labels, val_samples, val_labels, model, epochs)

    dumb_int_sequences_input, dumb_preds = create_embedding_layer(class_names, embedding_matrix, num_tokens, embedding_dim, True)
    dumb_model = create_model(dumb_int_sequences_input, dumb_preds)
    dumb_loss, dumb_acc, dumb_val_loss, dumb_val_acc = train_model(vectorizer, train_samples, train_labels, val_samples, val_labels, dumb_model, epochs)

    plot_dif_method(loss, val_loss, dumb_loss, dumb_val_loss)

main()
