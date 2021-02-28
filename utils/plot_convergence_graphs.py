import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def plot_convergences(train_acc_list, val_acc_list, train_loss_list, val_loss_list, model_name=None):
    plt.plot(list(train_acc_list), c="red", label ="train")
    plt.plot(list(val_acc_list), c="blue", label ="val")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    if model_name == None:
        plt.title('Accuracy per epoch- train & val')
    else:
        plt.title(f'{model_name} Accuracy per epoch- train & val')
    plt.legend()
    plt.show()

    plt.plot(list(train_loss_list), c="red", label ="train")
    plt.plot(list(val_loss_list), c="blue", label ="val")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    if model_name == None:
        plt.title('Loss per epoch- train & val')
    else:
        plt.title(f'{model_name} Loss per epoch- train & val')
    plt.legend()
    plt.show()


