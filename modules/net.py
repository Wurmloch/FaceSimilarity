from keras.models import Sequential
from keras.layers import MaxPooling2D, Flatten, Conv2D, Activation, Dense, Dropout
from keras.utils import plot_model
from keras import callbacks
import matplotlib.pyplot as plt

from modules import img_util


tb_callback = callbacks.TensorBoard(log_dir='Graph', histogram_freq=0, write_graph=True, write_images=True)


def create_model(width, height, num_classes):
    #
    #  image dimensions
    #
    img_height = height
    img_width = width
    # greyscale
    img_channels = 1

    input_shape = (img_height, img_width, img_channels)
    net_model = build_sequential(input_shape, num_classes=num_classes)

    net_model.summary()
    return net_model


def build_sequential(input_shape, num_classes):
    """
    builds the sequential model and returns it for compilation
    :param input_shape:
    :param num_classes:
    :return:
    """
    print("input shape for model: ", input_shape)
    n_model = Sequential()

    # first set of CONV => RELU => POOL
    n_model.add(Conv2D(32, kernel_size=(5, 5), padding="same", input_shape=input_shape))
    n_model.add(Activation("relu"))
    n_model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    n_model.add(Dropout(0.2))

    # second set of CONV => RELU => POOL
    n_model.add(Conv2D(64, kernel_size=(3, 3), padding="same"))
    n_model.add(Activation("relu"))
    n_model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    n_model.add(Dropout(0.2))

    # third set of CONV => RELU => POOL
    n_model.add(Conv2D(128, kernel_size=(3, 3), padding="same"))
    n_model.add(Activation("relu"))
    n_model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
    n_model.add(Dropout(0.2))

    # set of FC => RELU layers
    n_model.add(Flatten())
    n_model.add(Dense(500))
    n_model.add(Dropout(0.4))
    n_model.add(Activation("relu"))

    # softmax classifier
    n_model.add(Dense(num_classes))
    n_model.add(Activation("softmax"))

    """ For 1D data
    n_model.add(Dense(512, input_shape=input_shape))
    n_model.add(Activation('relu'))
    n_model.add(Dropout(0.2))
    n_model.add(MaxPooling2D(pool_size=(2, 2)))
    n_model.add(Dense(512))
    n_model.add(Activation('relu'))
    n_model.add(Dropout(0.2))
    n_model.add(MaxPooling2D(pool_size=(2, 2)))
    n_model.add(Dense(num_classes))
    n_model.add(Activation('softmax'))
    """

    return n_model


def show_training_graph(t_model_hist):
    """
    shows the graph of the training phase
    :param t_model_hist:
    :return:
    """
    # Get training and test loss histories
    training_loss = t_model_hist.history['loss']
    test_loss = t_model_hist.history['val_loss']
    acc = t_model_hist.history['acc']
    test_acc = t_model_hist.history['val_acc']

    # Create count of the number of epochs
    epoch_count = range(1, len(training_loss) + 1)

    # Visualize loss history
    ax1 = plt.gca()
    ax2 = ax1.twinx()

    plt.axis('normal')
    ax1.plot(epoch_count, training_loss, 'r--')
    ax1.plot(epoch_count, test_loss, 'y--')
    ax2.plot(epoch_count, acc, 'g-')
    ax2.plot(epoch_count, test_acc, 'b-')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('Loss')
    ax2.set_ylabel('Accuracy')
    ax1.legend(['Training Loss', 'Test Loss'])
    ax2.legend(['Training Accuracy', 'Test Accuracy'])
    plt.show()


def train(t_model, t_train_data, t_test_data, t_train_labels, t_test_labels):
    """
    trains a new model
    :param t_model:
    :param t_train_data:
    :param t_test_data:
    :param t_train_labels:
    :param t_test_labels:
    :return:
    """
    number_epochs = int(input('Number of epochs:\n--> '))
    batch_size = int(input('Batch size:\n--> '))
    print('training on model...')

    model_history = t_model.fit(
        x=t_train_data,
        y=t_train_labels,
        batch_size=batch_size,
        epochs=number_epochs,
        verbose=1,
        validation_data=(t_test_data, t_test_labels),
        callbacks=[tb_callback]
    )
    show_training_graph(model_history)

    print('evaluating model on validation generator...')
    (loss, accuracy) = t_model.evaluate(t_test_data, t_test_labels, batch_size=batch_size, verbose=1)
    print(t_model.metrics_names)

    print('\nTesting loss: {}\nTesting accuracy: {}\n'.format(loss, accuracy))

    model_json = t_model.to_json()
    with open("model.json", "w+") as json_file:
        json_file.write(model_json)
    t_model.save_weights("model_weights")
    t_model.save("model_full")
    return t_model


def load_train(lt_model, lt_train_data, lt_test_data, lt_train_labels, lt_test_labels):
    """
    continues training on a model with saved weights
    :param lt_model:
    :param lt_train_data:
    :param lt_test_data:
    :param lt_train_labels:
    :param lt_test_labels:
    :return:
    """
    print("loading model weights...")
    lt_model.load_weights("model_weights")
    print("model weights loaded.")
    number_epochs = int(input('Number of epochs:\n--> '))
    batch_size = int(input('Batch size:\n--> '))
    print('training on model...')

    model_history = lt_model.fit(
        x=lt_train_data,
        y=lt_train_labels,
        batch_size=batch_size,
        epochs=number_epochs,
        verbose=1,
        validation_data=(lt_test_data, lt_test_labels),
        callbacks=[tb_callback]
    )
    show_training_graph(model_history)

    print('evaluating model on validation generator...')
    (loss, accuracy) = lt_model.evaluate(lt_test_data, lt_test_labels, batch_size=batch_size, verbose=1)
    print(lt_model.metrics_names)

    print('\nTesting loss: {}\nTesting accuracy: {}\n'.format(loss, accuracy))

    model_json = lt_model.to_json()
    with open("model.json", "w+") as json_file:
        json_file.write(model_json)
    lt_model.save_weights("model_weights")
    lt_model.save("model_full")
    return lt_model


def load(l_model):
    """
    loads the model weights and returns the model
    :param l_model:
    :return:
    """
    print("loading model weights...")
    l_model.load_weights("model_weights")
    print("model weights loaded.")
    return l_model


def init_net():
    """
    initializes the NN --> TODO: keine normalisierten Daten laden bei init
    :return:
    """
    global model, train_data, test_data, train_labels, test_labels
    norm_dir = input('Enter the root folder of NORMALIZED images:\n--> ')
    train_data, test_data, train_labels, test_labels, num_classes = img_util.norm_image_generator(norm_dir)

    model = create_model(128, 128, num_classes=num_classes)
    model.compile(
        optimizer='sgd',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    plot_model(model)

    print('model created and compiled: ' + str(model))

    model_decision = int(input("train weights from new (1)\nload weights and continue training (2)"
                               "\nload weights without training (3):\n--> "))
    if model_decision == 1:
        model = train(model, train_data, test_data, train_labels, test_labels)
    elif model_decision == 2:
        model = load_train(model, train_data, test_data, train_labels, test_labels)
    elif model_decision == 3:
        model = load(model)

    return model, train_data, test_data, train_labels, test_labels


def get_net_values():
    """
    if model is already defined, return the globals
    :return:
    """
    return model, train_data, test_data, train_labels, test_labels
