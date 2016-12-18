import sys
import time
import argparse
import re
import os
import os.path
from PyQt4.Qt import *
from pybrain.tools.shortcuts import buildNetwork
from pybrain.datasets import ClassificationDataSet
from pybrain.structure.modules import SigmoidLayer
from pybrain.supervised.trainers import BackpropTrainer
from pybrain.utilities import percentError
from pybrain.tools.customxml.networkwriter import NetworkWriter
from pybrain.tools.customxml.networkreader import NetworkReader
from pybrain.supervised.trainers import RPropMinusTrainer

data_dir = ""


def init_brain(learn_data, epochs, TrainerClass=BackpropTrainer):
    global data_dir
    print("\t Epochs: ", epochs)
    if learn_data is None:
        return None
    print("Building network")
    net = buildNetwork(7 * 7, 7, 4, hiddenclass=SigmoidLayer)
    # net = buildNetwork(64 * 64, 32 * 32, 8 * 8, 5)
    # net = buildNetwork(64 * 64, 5, hiddenclass=LinearLayer)
    # fill dataset with learn data
    trans = {
        '0': 0, '1': 1, '2': 2, '3': 3
    }
    ds = ClassificationDataSet(7 * 7, nb_classes=4, class_labels=['0', '1', '2', '3'])
    for inp, out in learn_data:
        ds.appendLinked(inp, [trans[out]])
    ds.calculateStatistics()
    print("\tNumber of classes in dataset = {0}".format(ds.nClasses))
    print("\tOutput in dataset is ", ds.getField('target').transpose())
    ds._convertToOneOfMany(bounds=[0, 1])
    print("\tBut after convert output in dataset is \n", ds.getField('target'))
    trainer = TrainerClass(net, learningrate=0.1, verbose=True)
    trainer.setData(ds)
    print("\tEverything is ready for learning.\nPlease wait, training in progress...")
    start = time.time()
    trainer.trainEpochs(epochs=epochs)
    end = time.time()

    f = open(data_dir + "values.txt", "w")
    f.write("Training time: %.2f \n" % (end - start))
    f.write("Total epochs: %s \n" % (trainer.totalepochs))
    # f.write("Error: %.22f" % (trainer.trainingErrors[len(trainer.trainingErrors) - 1]))
    f.close()

    print("Percent of error: ", percentError(trainer.testOnClassData(), ds['class']))
    print("\tOk. We have trained our network.")
    NetworkWriter.writeToFile(net, data_dir + "net.xml")
    return net


def loadData(dir_name):
    list_dir = os.listdir(dir_name)
    list_dir.sort()
    list_for_return = []
    print("Loading data...")
    for filename in list_dir:
        out = [None, None]
        print("Working at {0}".format(dir_name + filename))
        print("\tTrying get number name.")
        num = re.search("(\d)-\w\.png", dir_name + filename)
        if num is None:
            print("\tFilename not matches pattern.")
            continue
        else:
            print("\tFilename matches! number is '{0}'. Appending...".format(num.group(1)))
            out[1] = num.group(1)
        print("\tTrying get number picture.")
        out[0] = get_data(dir_name + filename)
        print("\tChecking data size.")
        if len(out[0]) == 7 * 7:
            print("\tSize is ok.")
            list_for_return.append(out)
            print("\tInput data appended. All done!")
        else:
            print("\tData size is wrong. Skipping...")
    return list_for_return


def get_data(png_file):
    img = QImage(7, 7, QImage.Format_RGB32)
    data = []
    if img.load(png_file):
        for x in range(7):
            for y in range(7):
                data.append(qGray(img.pixel(x, y)) / 255.0)
    else:
        print("img.load({0}) failed!".format(png_file))
    return data


def work_brain(net, inputs):
    rez = net.activate(inputs)
    idx = 0
    data = rez[0]
    for i in range(1, len(rez)):
        if rez[i] > data:
            idx = i
            data = rez[i]
    return (idx, data, rez)


def test_brain(net, test_data):
    for data, right_out in test_data:
        out, rez, output = work_brain(net, data)
        print("For '{0}' our net said that it is '{1}'. Raw = {2}".format(right_out, "0123"[out], output))
    pass


def main():
    global data_dir
    app = QApplication([])
    p = argparse.ArgumentParser(description='PyBrain example')
    p.add_argument('-d', '--data-dir', default="./", help="Path to dir, containing data")
    p.add_argument('-e', '--epochs', default="1000",
                   help="Number of epochs for teach, use 0 for learning until convergence")
    args = p.parse_args()
    data_dir = args.data_dir
    learn_path = os.path.abspath(data_dir) + "/learn/"
    test_path = os.path.abspath(data_dir) + "/test/"
    if not os.path.exists(learn_path):
        print("Error: Learn directory not exists!")
        sys.exit(1)
    if not os.path.exists(test_path):
        print("Error: Test directory not exists!")
        sys.exit(1)
    learn_data = loadData(learn_path)
    test_data = loadData(test_path)
    # net = init_brain(learn_data, int(args.epochs), TrainerClass=RPropMinusTrainer)
    try:
        net = NetworkReader.readFrom(data_dir + "net.xml")
    except FileNotFoundError:
        print("Create net")
        net = init_brain(learn_data, int(args.epochs), TrainerClass=BackpropTrainer)
    print("Now we get working network. Let's try to use it on learn_data.")
    print("Here comes a tests on learn-data!")
    test_brain(net, learn_data)
    print("Here comes a tests on test-data!")
    test_brain(net, test_data)
    return 0


if __name__ == "__main__":
    sys.exit(main())
