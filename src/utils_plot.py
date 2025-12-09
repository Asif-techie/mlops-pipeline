# ============================
# File: src/utils_plot.py
# ============================

# src/utils_plot.py
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_curve, auc

def save_cm(y_true, y_pred, path):
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots()
    ax.imshow(cm, cmap='Blues')
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, cm[i,j], ha='center', va='center', color='red')
    ax.set_title("Confusion Matrix")
    fig.savefig(path)
    plt.close(fig)

def save_roc(y_true, y_score, path):
    fpr, tpr, _ = roc_curve(y_true, y_score)
    roc_auc = auc(fpr, tpr)
    fig, ax = plt.subplots()
    ax.plot(fpr, tpr, label=f"AUC={roc_auc:.3f}")
    ax.plot([0,1],[0,1],'--')
    ax.set_title("ROC Curve")
    ax.legend()
    fig.savefig(path)
    plt.close(fig)
