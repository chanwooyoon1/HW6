# This code is created by Chanwoo Yoon ad Juwon Lee
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
# part3
#-----------------------------------------------------------------------------------------------------#
# create Movie data set
def create_parent_directory_pt3():
    data_path = keras.utils.get_file(
        "aclImdb_v1.tar.gz",
        "https://ai.stanford.edu/~amaas/data/sentiment/aclImdb_v1.tar.gz",
        untar=True
    )
    data_dir = pathlib.Path(data_path).parent / "aclImdb"
    return data_dir

def data2list_pt3(data_dir):
    samples = []
    labels = []
    class_names = ["neg", "pos"]
    for class_index, class_name in enumerate(class_names):
        class_dir = data_dir / "train" / class_name
        fnames = os.listdir(class_dir)
        for fname in fnames:
            fpath = class_dir / fname
            with open(fpath, encoding="latin-1") as f:
                content = f.read()
                samples.append(content)
                labels.append(class_index)
    return samples, labels, class_names
#-----------------------------------------------------------------------------------------------------#

# create newspaper data set
def create_parent_directory():
    data_path = keras.utils.get_file(
        "news20.tar.gz",
        "http://www.cs.cmu.edu/afs/cs.cmu.edu/project/theo-20/www/data/news20.tar.gz",
        untar=True,
    )
    data_dir = pathlib.Path(data_path).parent / "20_newsgroup"
    return data_dir

# convert data set to processable lists
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

# shuffle data
def shuffle_data(samples, labels):
    seed = 1337
    rng = np.random.RandomState(seed)
    rng.shuffle(samples)
    rng = np.random.RandomState(seed)
    rng.shuffle(labels)
    return samples, labels
    
# get training and validation sets
def get_trainNval_data(samples, labels, val_position, validation_split=0.2):
    # val_position controls where to set as a validation set within the training data
    num_validation_samples = int(validation_split * len(samples))
    val_start_index = val_position * num_validation_samples
    val_end_index = val_start_index + num_validation_samples

    val_samples = samples[val_start_index:val_end_index]
    val_labels = labels[val_start_index:val_end_index]

    train_samples = samples[:val_start_index] + samples[val_end_index:]
    train_labels = labels[:val_start_index] + labels[val_end_index:]

    return train_samples, val_samples, train_labels, val_labels

# make words to numbers
def vectorize_data(train_samples, val_samples):
    vectorizer = layers.TextVectorization(max_tokens=20000, output_sequence_length=200)
    text_ds = tf_data.Dataset.from_tensor_slices(train_samples).batch(128)
    vectorizer.adapt(text_ds)
    voc = vectorizer.get_vocabulary()
    word_index = dict(zip(voc, range(len(voc))))
    path_to_glove_file = '/content/glove.6B.100d.txt'
    embeddings_index = {}
    with open(path_to_glove_file) as f:
        for line in f:
            word, coefs = line.split(maxsplit=1)
            coefs = np.fromstring(coefs, "f", sep=" ")
            a = coefs
            embeddings_index[word] = coefs
    return vectorizer, voc, word_index, embeddings_index

# merge embedding vectors into a matrix
def create_embedding_matrix(word_index, embeddings_index, num_tokens, embedding_dim):
    hits = 0
    misses = 0
    # Prepare embedding matrix
    embedding_matrix = np.zeros((num_tokens, embedding_dim))
    for word, i in word_index.items():
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # Words not found in embedding index will be all-zeros
            embedding_matrix[i] = embedding_vector
            hits += 1
        else:
            misses += 1
    return embedding_matrix

# creates a model
def create_model(int_sequences_input, preds):
    model = keras.Model(int_sequences_input, preds)
    model.summary()
    return model

# train a model and returns losses and acuracies
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

# creates an embedding layer
def create_embedding_layer(class_names, embedding_matrix, num_tokens, embedding_dim, dumb):
    # if dumb is True, then this function changes embedding_matrix to one-hot one
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

# plots results(not for k folds)
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

# plots results from GloVE and dumb embedding
def plot_dif_method(train, val, dumb_train, dumb_val, valuetype, trainability):
    train_range = range(len(train))
    val_range = range(len(val))
    plt.plot(train_range, train, color = 'b', label='GloVE training ' + valuetype)
    plt.plot(val_range, val, color = 'b', linestyle='--', label='GloVE validation ' + valuetype)
    plt.plot(train_range, dumb_train, color = 'r', label='Dumb training ' + valuetype)
    plt.plot(val_range, dumb_val, color = 'r', linestyle='--', label='Dumb validation ' + valuetype)
    plt.title('Training and Validation ' + valuetype + ' (' + trainability + ')')
    plt.legend(loc='upper left', fontsize='small')
    plt.xlabel('Epochs')
    plt.ylabel(valuetype)
    plt.show()

# puts the above function together and returns losses and accuracies
def all_together(train_samples, val_samples, train_labels, val_labels, class_names, epochs, num_tokens, embedding_dim):
    vectorizer, voc, word_index, embeddings_index = vectorize_data(train_samples, val_samples)
    embedding_matrix = create_embedding_matrix(word_index, embeddings_index, num_tokens, embedding_dim)
    int_sequences_input, preds = create_embedding_layer(class_names, embedding_matrix, num_tokens, embedding_dim, False)
    model = create_model(int_sequences_input, preds)
    # GloVE
    int_sequences_input, preds = create_embedding_layer(class_names, embedding_matrix, num_tokens, embedding_dim, False)
    model = create_model(int_sequences_input, preds)
    loss, acc, val_loss, val_acc = train_model(vectorizer, train_samples, train_labels, val_samples, val_labels, model, epochs)
    # Dumb
    dumb_int_sequences_input, dumb_preds = create_embedding_layer(class_names, embedding_matrix, num_tokens, embedding_dim, True)
    dumb_model = create_model(dumb_int_sequences_input, dumb_preds)
    dumb_loss, dumb_acc, dumb_val_loss, dumb_val_acc = train_model(vectorizer, train_samples, train_labels, val_samples, val_labels, dumb_model, epochs)
    return [[loss, val_loss, dumb_loss, dumb_val_loss], [acc, val_acc, dumb_acc, dumb_val_acc]]

# prints numeric data for every epoch
def get_numeric_data(alist, valuetype):
    epochs_range = range(len(alist[0]))
    avg_train = []
    avg_val = []
    avg_dumb_train = []
    avg_dumb_val = []
    std_train = []
    std_val = []
    std_dumb_train = []
    std_dumb_val = []
    for i in range(len(alist[0])):
        avg_train.append(sum(alist[0][i])/len(alist[0][i]))
        avg_val.append(sum(alist[1][i])/len(alist[1][i]))
        avg_dumb_train.append(sum(alist[2][i])/len(alist[2][i]))
        avg_dumb_val.append(sum(alist[3][i])/len(alist[3][i]))
        std_train.append(np.std(alist[0][i]))
        std_val.append(np.std(alist[1][i]))
        std_dumb_train.append(np.std(alist[2][i]))
        std_dumb_val.append(np.std(alist[3][i]))
    print('GloVE training ' + valuetype)
    print(avg_train)
    print('GloVE validation ' + valuetype)
    print(avg_val)
    print('GloVE training SEM')
    print(std_train)
    print('GloVE validation SEM')
    print(std_val)
    print('Dumb training ' + valuetype)
    print(avg_dumb_train)
    print('Dumb validation ' + valuetype)
    print(avg_dumb_val)
    print('Dumb training SEM')
    print(std_dumb_train)
    print('Dumb validation SEM')
    print(std_dumb_val)

# plots results(for k folds)
def plot_kfolds(alist, valuetype):
    epochs_range = range(len(alist[0]))
    avg_train = []
    avg_val = []
    avg_dumb_train = []
    avg_dumb_val = []
    std_train = []
    std_val = []
    std_dumb_train = []
    std_dumb_val = []
    for i in range(len(alist[0])):
        avg_train.append(sum(alist[0][i])/len(alist[0][i]))
        avg_val.append(sum(alist[1][i])/len(alist[1][i]))
        avg_dumb_train.append(sum(alist[2][i])/len(alist[2][i]))
        avg_dumb_val.append(sum(alist[3][i])/len(alist[3][i]))
        std_train.append(np.std(alist[0][i]))
        std_val.append(np.std(alist[1][i]))
        std_dumb_train.append(np.std(alist[2][i]))
        std_dumb_val.append(np.std(alist[3][i]))
    train = range(len(avg_train))
    val = range(len(avg_val))
    dumb_train = range(len(avg_dumb_train))
    dumb_val = range(len(avg_dumb_val))
    plt.plot(train, avg_train, color = 'b', label = 'GloVE training ' + valuetype)
    plt.plot(val, avg_val, color = 'b', linestyle = '--', label = 'GloVE validation ' + valuetype)
    plt.plot(dumb_train, avg_dumb_train, color = 'r', label = 'Dumb training ' + valuetype)
    plt.plot(dumb_val, avg_dumb_val, color = 'r', linestyle = '--', label = 'Dumb validation ' + valuetype)

    plt.fill_between(
        epochs_range,
        [avg_train[i] - std_train[i] for i in range(len(std_train))],
        [avg_train[i] + std_train[i] for i in range(len(std_train))],
        color="blue",
        alpha=0.2
    )
    plt.fill_between(
        epochs_range,
        [avg_val[i] - std_val[i] for i in range(len(std_val))],
        [avg_val[i] + std_val[i] for i in range(len(std_val))],
        color="blue",
        alpha=0.2
    )

    plt.fill_between(
        epochs_range,
        [avg_dumb_train[i] - std_dumb_train[i] for i in range(len(std_dumb_train))],
        [avg_dumb_train[i] + std_dumb_train[i] for i in range(len(std_dumb_train))],
        color="red",
        alpha=0.2
    )
    plt.fill_between(
        epochs_range,
        [avg_dumb_val[i] - std_dumb_val[i] for i in range(len(std_dumb_val))],
        [avg_dumb_val[i] + std_dumb_val[i] for i in range(len(std_dumb_val))],
        color="red",
        alpha=0.2
    )

    plt.title('Training and Validation ' + valuetype + ' k(5)-fold (trainable) on IMDB Movies Dataset')
    plt.legend(loc='upper left', fontsize='small')
    plt.xlabel('Epochs')
    plt.ylabel(valuetype)
    plt.show()

# creates three sets of for k(5) fold
def kfolds_main():
    data_dir = create_parent_directory()
    epochs = 20
    samples, labels, class_names = data2list(data_dir)
    samples, labels = shuffle_data(samples, labels)
    num_tokens = 20002
    embedding_dim = 100

    # create 3 different sets of data sets for k fold (k=5)
    #-----------------------------------------------------------------------------------------------#
    train_samples1, val_samples1, train_labels1, val_labels1 = get_trainNval_data(samples, labels, 0)
    train_samples2, val_samples2, train_labels2, val_labels2 = get_trainNval_data(samples, labels, 1)
    train_samples3, val_samples3, train_labels3, val_labels3 = get_trainNval_data(samples, labels, 2)
    #-----------------------------------------------------------------------------------------------#

    all_list1 = all_together(train_samples1, val_samples1, train_labels1, val_labels1, class_names, epochs, num_tokens, embedding_dim)
    all_list2 = all_together(train_samples2, val_samples2, train_labels2, val_labels2, class_names, epochs, num_tokens, embedding_dim)
    all_list3 = all_together(train_samples3, val_samples3, train_labels3, val_labels3, class_names, epochs, num_tokens, embedding_dim)

    all_lists = [all_list1, all_list2, all_list3]

    all_list_final = []
    for i in range(2):
        sublist = []
        for all_list in all_lists:
            sublist.append(all_list[i])
        all_list_final.append(sublist)
    integrated_list = []
    for p in range(2):
        alist = []
        for i in range(len(all_list_final[p][0])):
            sublist = []
            for j in range(len(all_list_final[p][0][i])):
                combined = [all_list_final[p][k][i][j] for k in range(len(all_list_final[p]))]
                sublist.append(combined)
            alist.append(sublist)
        integrated_list.append(alist)

    get_numeric_data(integrated_list[0], 'loss')
    print('\n')
    get_numeric_data(integrated_list[1], 'accuracy')

    plot_kfolds(integrated_list[0], 'loss')
    plot_kfolds(integrated_list[1], 'accuracy')

kfolds_main()

def main():
    data_dir = create_parent_directory_pt3()
    epochs = 20
    samples, labels, class_names = data2list_pt3(data_dir)
    samples, labels = shuffle_data(samples, labels)
    train_samples, val_samples, train_labels, val_labels = get_trainNval_data(samples, labels, 0)
    vectorizer, voc, word_index, embeddings_index = vectorize_data(train_samples, val_samples)

    num_tokens = len(voc) + 2
    print(num_tokens)
    embedding_dim = 100

    embedding_matrix = create_embedding_matrix(word_index, embeddings_index, num_tokens, embedding_dim)

    int_sequences_input, preds = create_embedding_layer(class_names, embedding_matrix, num_tokens, embedding_dim, False)
    model = create_model(int_sequences_input, preds)
    loss, acc, val_loss, val_acc = train_model(vectorizer, train_samples, train_labels, val_samples, val_labels, model, epochs)

    dumb_int_sequences_input, dumb_preds = create_embedding_layer(class_names, embedding_matrix, num_tokens, embedding_dim, True)
    dumb_model = create_model(dumb_int_sequences_input, dumb_preds)
    dumb_loss, dumb_acc, dumb_val_loss, dumb_val_acc = train_model(vectorizer, train_samples, train_labels, val_samples, val_labels, dumb_model, epochs)
    for i in range(len(loss)):
        print("Epoch" + str(i+1))
        print("GloVE model:")
        print("Training loss-> " + str(loss[i]) + "  " + "Training accuracy-> " + str(acc[i]))
        print("Validation loss-> " + str(val_loss[i]) + "  " + "Validation accuracy-> " + str(val_acc[i]) + "\n")
        print("Dumb model:")
        print("Training loss-> " + str(dumb_loss[i]) + "  " + "Training accuracy-> " + str(dumb_acc[i]))
        print("Validation loss-> " + str(dumb_val_loss[i]) + "  " + "Validation accuracy-> " + str(dumb_val_acc[i]) + "\n" + "\n")

    print("GloVE model:")
    print("Average training loss-> " + str(np.mean(loss)) + "  " + "Average training accuracy-> " + str(np.mean(acc)))
    print("SEM of training loss->" + str(np.std(loss)) + "  " + "SEM of training accuracy-> " + str(np.std(acc)))
    print("Average validation loss-> " + str(np.mean(val_loss)) + "  " + "Average validation accuracy-> " + str(np.mean(val_acc)))
    print("SEM of validation loss->" + str(np.std(val_loss)) + "  " + "SEM of validation accuracy-> " + str(np.std(val_acc)) + "\n")

    print("Dumb model:")
    print("Average training loss-> " + str(np.mean(dumb_loss)) + "  " + "Average training accuracy-> " + str(np.mean(dumb_acc)))
    print("SEM of training loss->" + str(np.std(dumb_loss)) + "  " + "SEM of training accuracy-> " + str(np.std(dumb_acc)))
    print("Average validation loss-> " + str(np.mean(dumb_val_loss)) + "  " + "Average validation accuracy-> " + str(np.mean(dumb_val_acc)))
    print("SEM of validation loss->" + str(np.std(dumb_val_loss)) + "  " + "SEM of validation accuracy-> " + str(np.std(dumb_val_acc)))
    plot_dif_method(loss, val_loss, dumb_loss, dumb_val_loss, 'loss', 'trainable')
    plot_dif_method(acc, val_acc, dumb_acc, dumb_val_acc, 'accuracy', 'trainable')

