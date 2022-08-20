from sklearn.metrics import confusion_matrix
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt

import os

cwd = os.path.dirname(os.path.abspath(__file__))+'/'
cwd = cwd.replace('\\','/')


def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
        print("Folder created")
    else:
        print("Folder already exists")


def main(Normalize = True):
    
    # set font to Times New Roman
    plt.rcParams['font.family'] = 'Times New Roman' 

    # load confusion matrixs
    # categories = ['1', '2', '3', '4', '7']
    categories=['animals','chairs','human faces','fruits','vehicles']
    
    reports_path = cwd+"Archive/Subject wise/reports"
    subjects = np.sort(np.array(os.listdir(reports_path), dtype = int))

    save_dir = cwd+"confusion_matrixs/"
    create_folder(save_dir)

    for i, subject in enumerate(subjects):
        print(f"\nConfusion Matrix subject {subject}")
        cm = np.loadtxt(reports_path+"/"+str(subject)+"/confusion_matrix.txt", dtype=int)
        print(cm)

        if Normalize:
            cm = cm/np.sum(cm)

            sns.heatmap(cm, annot=True, 
                        fmt='.2%', cmap='Blues',
                        xticklabels=categories,
                        yticklabels=categories)
        else:
            sns.heatmap(cm, annot=True, 
                        fmt='d', cmap='Blues',
                        xticklabels=categories,
                        yticklabels=categories)
        plt.title(f"Confusion Matrix Subject {subject}")
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()
        # plt.show()
        plt.savefig(save_dir+str(subject)+".svg")
        plt.close()


if __name__ == "__main__":
    main(Normalize=True)
