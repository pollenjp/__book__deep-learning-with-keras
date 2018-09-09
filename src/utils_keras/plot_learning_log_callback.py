import matplotlib.pyplot as plt
import keras
#from IPython.display import clear_output

class PlotLearningLogCallback(keras.callbacks.Callback):
    def on_train_begin(self, logs={}):
        self.i = 0
        self.x = []
        self.losses = []
        self.val_losses = []
        self.acc = []
        self.val_acc = []
        self.fig = plt.figure()
        self.logs = []
        #self.f, (self.ax1, self.ax2) = plt.subplots(1, 2, sharex=True)
        self.max_epoch = self.params["epochs"]

    def on_epoch_end(self, epoch, logs={}):
        self.logs.append(logs)
        self.x.append(self.i)
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.acc.append(logs.get('acc'))
        self.val_acc.append(logs.get('val_acc'))
        self.i += 1

        #f, (ax1, ax2) = plt.subplots(1, 2, sharex=True)
        fig = plt.figure(figsize=(15, 5))
        #clear_output(wait=True)
        val_list = [("loss", self.losses, self.val_losses),
                    ("acc",  self.acc,    self.val_acc)]
        for idx, (label, val1, val2) in enumerate(val_list, start=1):
            ax = fig.add_subplot(1, 2, idx)
            if label == "loss":
                ax.set_yscale('log')
            ax.plot(self.x, val1, label=label)
            ax.plot(self.x, val2, label="val_{}".format(label))
            ax.set_xlim(left=-0.5, right=self.max_epoch)
            ax.legend()
        plt.show()

