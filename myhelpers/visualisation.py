import itertools
import matplotlib.pyplot as plt
from IPython.display import display
import numpy as np
from sklearn.metrics import confusion_matrix
from matplotlib import style

style.use('fivethirtyeight')


def plot_hist(feature, a, b, la, lb, bins=20):
    x1 = np.array(a[feature].dropna())
    x2 = np.array(b[feature].dropna())
    plt.hist([x1, x2], label=[la, lb], bins=bins, color=['r', 'b'])
    plt.legend(loc="upper left")
    plt.title('distribution relative de %s' %feature)
    plt.show()

def plot_roc(models, ax=None):
    if ax is None:
        fig, ax = plt.subplots()
    for i, model in enumerate(models):
        if model['fpr'] is not None:
            if i==0: 
                lw = 3
            else:
                lw = 1
            ax.plot(model['fpr'], model['tpr'], lw=lw,
                    label='{} AUC = {:0.2f}'.format(model['name'], model['roc_auc']))
    ax.set_title('Receiver Operating Characteristic')
    ax.plot([0,1],[0,1],'k--')
    ax.set_xlim([-0.01,1])
    ax.set_ylim([0,1.01])
    ax.legend(loc='lower right')
    ax.set_ylabel('True Positive Rate')
    ax.set_xlabel('False Positive Rate')
    return


def plot_confusion_matrix(cm, classes,
                          reverse=True,
                          ax=None, fig=None,
                          normalize=False,
                          title='Confusion matrix',
                          colorbar=True):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if reverse is True:
        cm = cm[::-1,::-1]
        classes = classes[::-1]
    ax.set_title(title)
    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticks(tick_marks)
    ax.set_yticklabels(classes)
    cmap = plt.cm.Blues

    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    if colorbar:
        fig.colorbar(im, ax=ax, shrink=0.7)

    if normalize:
        cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        txt = "{:.0f}".format(cm[i,j])
        if normalize:
            txt = txt + "\n{:0.1%}".format(cm_norm[i,j])

        ax.text(j, i, txt, fontsize=14, fontweight='bold',
                 horizontalalignment="center", verticalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    ax.grid('off')
    ax.set_ylabel('True label')
    ax.set_xlabel('Predicted label')
    return


def visualisation(y_test, y_pred, model=None, show_roc=False, show_cm=True):

    n_axes = sum([show_roc, show_cm])

    if n_axes > 0:
        fig, axes = plt.subplots(1,n_axes, figsize=(5*n_axes,4))
        i = 0

        if n_axes == 1:
            axes = [axes]
            # Plot ROC curve
        if show_roc:
            ax = axes[i]
            plot_roc(model, ax)
            i += 1

        if show_cm:
            ax = axes[i]
            conf_mat = confusion_matrix(y_test, y_pred)
            plot_confusion_matrix(conf_mat, 
                                classes=['Survived', 'Dead'],
                                reverse=True,
                                normalize=True,
                                ax=ax, 
                                fig=fig)

        fig.tight_layout()
        plt.show()