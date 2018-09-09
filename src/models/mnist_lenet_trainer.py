import os
import keras
from keras.optimizers import Adam
from keras.callbacks import TensorBoard

class MnistLenetTrainer():

    def __init__(self, model, loss, optimizer, metrics=["accuracy"], logdir = "logdir_lenet"):
        self.model = model
        self.model.compile( loss=loss, optimizer=optimizer, metrics=["accuracy"] )
        self.logdir = logdir

        if os.path.exists(self.logdir):
            import shutil
            shutil.rmtree(self.logdir)  # remove previous execution
        os.mkdir(self.logdir)


