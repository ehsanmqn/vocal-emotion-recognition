
from keras.utils import np_utils

from speechemotionrecognition.dnn import LSTM, CNN
from speechemotionrecognition.mlmodel import NN, SVM, RF
from speechemotionrecognition.utilities import get_data, class_labels, read_file

dataset_path = 'Mixture'


def dnn_train():

    # Prepare data
    x_train, x_test, y_train, y_test = get_data(dataset_path=dataset_path, flatten=False)
    y_train = np_utils.to_categorical(y_train)
    y_test = np_utils.to_categorical(y_test)

    # Initial model
    print('Starting LSTM')
    model = LSTM(input_shape=x_train[0].shape, num_classes=len(class_labels))

    # Train model
    model.train(x_train, y_train, x_test, y_test)

    # Save traned model
    model.save_model()

    # Load model to use
    model.load_model("LSTM_best_model.h5")

    # Evaluate model
    model.evaluate(x_test, y_test)
    model.evaluate(x_train, y_train)

    # Predict with loaded model
    # model.predict(x_test)
    # model.predict_classes(x_test)


    # print 'LSTM Done\n Starting CNN'
    # in_shape = x_train[0].shape
    # x_train = x_train.reshape(x_train.shape[0], in_shape[0], in_shape[1], 1)
    # x_test = x_test.reshape(x_test.shape[0], in_shape[0], in_shape[1], 1)
    # model = CNN(input_shape=x_train[0].shape, num_classes=len(class_labels))
    # model.train(x_train, y_train, x_test, y_test)
    # model.evaluate(x_test, y_test)
    # model.evaluate(x_train, y_train)
    # print 'CNN Done'

def dnn_evaluate(model_file_name):

    # Prepare data
    x_train, x_test, y_train, y_test = get_data(dataset_path=dataset_path, flatten=False)
    y_test = np_utils.to_categorical(y_test)
    y_train = np_utils.to_categorical(y_train)

    # Initial model
    print('Starting LSTM')
    model = LSTM(input_shape=x_train[0].shape, num_classes=len(class_labels))

    # Load model to use
    model.load_model(model_file_name)

    # Evaluate model
    model.evaluate(x_train, y_train)
    model.evaluate(x_test, y_test)

    # Predict with loaded model
    # model.predict(x_test)
    # model.predict_classes(x_test)


def dnn_test(model_file_name, input_file_name):

    data = read_file(input_file_name)

    # Initial model
    print('Loading model ...')
    model = LSTM(input_shape=data[0].shape, num_classes=len(class_labels))

    # Load model to use
    model.load_model(model_file_name)

    # Predict with loaded model
    model.predict(data)
    print("Predict result: ", class_labels[model.predict_classes(data)[0]])


def ml_example():
    x_train, x_test, y_train, y_test = get_data(dataset_path=dataset_path)
    models = [NN, RF, SVM]
    for M in models:
        model = M()
        print('Starting', model.name)
        model.train(x_train, y_train)
        model.evaluate(x_test, y_test)
        print(model.name, 'Done')


if __name__ == "__main__":
    # ml_example()
    # dnn_train()
    # dnn_evaluate("LSTM_best_model.h5")
    dnn_test("LSTM_best_model.h5", "tests/neutral1.wav")
