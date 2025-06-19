import numpy as np
from matplotlib import pyplot as plt
from prettytable import PrettyTable
from sklearn.metrics import roc_curve
from itertools import cycle

from Image_Results import *


def Statistical(data):
    Min = np.min(data)
    Max = np.max(data)
    Mean = np.mean(data)
    Median = np.median(data)
    Std = np.std(data)
    return np.asarray([Min, Max, Mean, Median, Std])


def plotConvResults():
    # matplotlib.use('TkAgg')
    Fitness = np.load('Fitness.npy', allow_pickle=True)
    Algorithm = ['TERMS', 'CO', 'GOA', 'WWPA', 'MA', 'PROPOSED']

    Terms = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD']
    Conv_Graph = np.zeros((5, 5))
    for j in range(5):  # for 5 algms
        Conv_Graph[j, :] = Statistical(Fitness[j, :])

    Table = PrettyTable()
    Table.add_column(Algorithm[0], Terms)
    for j in range(len(Algorithm) - 1):
        Table.add_column(Algorithm[j + 1], Conv_Graph[j, :])
    print('-------------------------------------------------- Statistical Analysis  ',
          '--------------------------------------------------')
    print(Table)

    fig = plt.figure()
    fig.canvas.manager.set_window_title('Convergence Curve')
    length = np.arange(50)
    Conv_Graph = Fitness
    plt.plot(length, Conv_Graph[0, :], color='r', linewidth=3, marker='*', markerfacecolor='red',
             markersize=12, label='CO-ODRDANet')
    plt.plot(length, Conv_Graph[1, :], color='g', linewidth=3, marker='*', markerfacecolor='green',
             markersize=12, label='GOA-ODRDANet')
    plt.plot(length, Conv_Graph[2, :], color='b', linewidth=3, marker='*', markerfacecolor='blue',
             markersize=12, label='WPA-ODRDANet')
    plt.plot(length, Conv_Graph[3, :], color='m', linewidth=3, marker='*', markerfacecolor='magenta',
             markersize=12, label='MA-ODRDANet')
    plt.plot(length, Conv_Graph[4, :], color='k', linewidth=3, marker='*', markerfacecolor='black',
             markersize=12, label='OMA-ODRDANet')
    plt.xlabel('No. of Iteration')
    plt.ylabel('Cost Function')
    plt.legend(loc=1)
    # plt.savefig("./Results/Conv.png")
    plt.show()


def Kfold_plot_results():
    eval = np.load('Eval_all_KFold.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1_score', 'MCC',
             'FOR', 'PT',
             'BA', 'FM', 'BM', 'MK', 'PLHR', 'lrminus', 'DOR', 'Prevalence', 'Threat Score']
    Graph_Terms = [1, 8, 10, 18]
    Kfold = [1, 2, 3, 4, 5]

    for i in range(eval.shape[0]):
        for j in range(len(Graph_Terms)):
            Graph = np.zeros(eval.shape[1:3])
            for k in range(eval.shape[1]):
                for l in range(eval.shape[2]):
                    if j == 9:
                        Graph[k, l] = eval[i, k, l, Graph_Terms[j] + 4]
                    else:
                        Graph[k, l] = eval[i, k, l, Graph_Terms[j] + 4]

            fig = plt.figure()
            fig.canvas.manager.set_window_title('KFold')
            plt.plot(Kfold, Graph[:, 0], color='#00FFFF', linewidth=3, marker='*',
                     markerfacecolor='#9A32CD', markersize=16,
                     label="CO-ODRDANet")
            plt.plot(Kfold, Graph[:, 1], color='#9A32CD', linewidth=3, marker='*',
                     markerfacecolor='#FF4500', markersize=16,
                     label="GOA-ODRDANet")
            plt.plot(Kfold, Graph[:, 2], color='#FF1493', linewidth=3, marker='*',
                     markerfacecolor='cyan', markersize=16,
                     label="WPA-ODRDANet")
            plt.plot(Kfold, Graph[:, 3], color='r', linewidth=3, marker='*',
                     markerfacecolor='#8B6969', markersize=16,
                     label="MA-ODRDANet")
            plt.plot(Kfold, Graph[:, 4], color='k', linewidth=3, marker='h', markerfacecolor='black',
                     markersize=12,
                     label="OMA-ODRDANet")
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.16),
                       ncol=3, fancybox=True, shadow=True)
            plt.xlabel('KFold')
            plt.xticks(Kfold, ('1', '2', '3', '4', '5'))
            plt.ylabel(Terms[Graph_Terms[j]])
            # path = "./Results/%s_Kfold_line.png" % (Terms[Graph_Terms[j]])
            # plt.savefig(path)
            plt.show()

            fig = plt.figure()
            ax = fig.add_axes([0.12, 0.12, 0.8, 0.8])
            fig.canvas.manager.set_window_title('KFold')
            X = np.arange(5)
            ax.bar(X + 0.00, Graph[:, 5], color='brown', width=0.10, label="Resnet")
            ax.bar(X + 0.10, Graph[:, 6], color='#069AF3', width=0.10, label="Shufflenet")
            ax.bar(X + 0.20, Graph[:, 7], color='pink', width=0.10, label="Densenet")
            ax.bar(X + 0.30, Graph[:, 8], color='#FF4500', width=0.10, label="DRDANet")
            ax.bar(X + 0.40, Graph[:, 4], color='k', width=0.10, label="OMA-ODRDANet")
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.13),
                       ncol=3, fancybox=True, shadow=True)
            plt.xticks(X + 0.20, ('1', '2', '3', '4', '5'))
            plt.xlabel('KFold')
            plt.ylabel(Terms[Graph_Terms[j]])
            # path = "./Results/%s_Kfold_bar.png" % (Terms[Graph_Terms[j]])
            # plt.savefig(path)
            plt.show()


def Plot_ROC_Curve():
    lw = 2
    cls = ['Resnet', 'Shufflenet', 'Densenet', 'DRDANet', 'OMA-ODRDANet']
    Actual = np.load('Targets.npy', allow_pickle=True)
    fig = plt.figure()
    fig.canvas.manager.set_window_title('ROC Curve')
    colors = cycle(["blue", "darkorange", "limegreen", "deeppink", "black"])
    for i, color in zip(range(5), colors):  # For all classifiers
        Predicted = np.load('Y_Score.npy', allow_pickle=True)[0][i]
        false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(Actual.ravel(), Predicted.ravel())
        plt.plot(
            false_positive_rate1,
            true_positive_rate1,
            color=color,
            lw=lw,
            label=cls[i])

    plt.plot([0, 1], [0, 1], "k--", lw=lw)
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.title('Accuracy')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curve")
    plt.legend(loc="lower right")
    # path = "./Results/ROC.png"
    # plt.savefig(path)
    plt.show()


def plot_seg_results():
    Eval_all = np.load('Eval_all1.npy', allow_pickle=True)
    Statistics = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD']
    Methods = ['TERMS', 'Unet', 'Unet++', 'ResUnet', 'DenseUnet', 'PROPOSED']
    Terms = ['Dice Coefficient', 'Jaccard', 'Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV',
             'FDR', 'F1-Score', 'MCC']

    for n in range(Eval_all.shape[0]):
        value_all = Eval_all[n, :]

        stats = np.zeros((value_all[0].shape[1] - 4, value_all.shape[0] + 4, 5))
        for i in range(4, value_all[0].shape[1] - 9):
            for j in range(value_all.shape[0] + 4):
                if j < value_all.shape[0]:
                    stats[i, j, 0] = np.max(value_all[j][:, i])
                    stats[i, j, 1] = np.min(value_all[j][:, i])
                    stats[i, j, 2] = np.mean(value_all[j][:, i])
                    stats[i, j, 3] = np.median(value_all[j][:, i])
                    stats[i, j, 4] = np.std(value_all[j][:, i])

            X = np.arange(1)
            name = [0, 0.2, 0.4, 0.6, 0.8]
            fig = plt.figure()
            fig.canvas.manager.set_window_title('Mean- ' + str(Terms[i - 4]))
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
            ax.bar(X + 0.00, stats[i, 0, 2:3], color='r', width=0.10, label="Unet")
            ax.bar(X + 0.20, stats[i, 1, 2:3], color='g', width=0.10, label="Unet++")
            ax.bar(X + 0.40, stats[i, 2, 2:3], color='b', width=0.10, label="ResUnet")
            ax.bar(X + 0.60, stats[i, 3, 2:3], color='m', width=0.10, label="DenseUnet")
            ax.bar(X + 0.80, stats[i, 4, 2:3], color='k', width=0.10, label="MFRCNet")
            plt.xticks(name, ('Unet', 'Unet++', 'ResUnet', 'DenseUnet', 'MFRCNet'))
            plt.ylabel(Terms[i - 4])
            # path = "./Results/Dataset_%s_%s_Mean_met.png" % (str(n + 1), Terms[i - 4])
            # plt.savefig(path)
            plt.show()


def Epoch_Table():
    eval = np.load('Eval_epoch.npy', allow_pickle=True)
    Terms = ['Accuracy', 'FOR', 'BA', 'FM', 'BM', 'Sensitivity', 'F1 Score', 'MCC', 'Specificity', 'Precision', 'FPR',
             'FNR', 'NPV',
             'FDR', 'PT',
             'MK', 'PLHR', 'lrminus', 'DOR', 'prevalence', 'TS']
    Algorithm = ['TERMS', 'CO-ODRDANet', 'GOA-ODRDANet', 'WPA-ODRDANet', 'MA-ODRDANet', 'OMA-ODRDANet']
    Classifier = ['TERMS', 'Resnet', 'Shufflenet', 'Densenet', 'DRDANet', 'OMA-ODRDANet']

    Epoch = [100, 200, 300, 400, 500]
    for k in range(eval.shape[1]):
        value1 = eval[0, k, :, 4:]

        Table = PrettyTable()
        Table.add_column(Algorithm[0], Terms[1:10])
        for j in range(len(Algorithm) - 1):
            Table.add_column(Algorithm[j + 1], value1[j, 1:10])
        print('-------------------------------------------------- ', Epoch[k], '-Epochs',
              '-Algorithm Comparison',
              '--------------------------------------------------')
        print(Table)

        Table = PrettyTable()
        Table.add_column(Classifier[0], Terms[1:10])
        for j in range(len(Classifier) - 1):
            Table.add_column(Classifier[j + 1], value1[len(Algorithm) + j - 1, 1:10])
        print('--------------------------------------------------', Epoch[k], '-Epochs ',
              '-Classifier Comparison',
              '--------------------------------------------------')
        print(Table)


def plot_Class_Results():
    eval = np.load('Eval_Class.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1_score', 'MCC',
             'FOR', 'PT',
             'BA', 'FM', 'BM', 'MK', 'PLHR', 'lrminus', 'DOR', 'Prevalence', 'Threat Score']
    Graph_Terms = [0]
    Kfold = ['bacterial_leaf_blight', 'bacterial_leaf_streak', 'bacterial_panicle_blight', 'blast', 'brown_spot']

    for i in range(eval.shape[0]):
        for j in range(len(Graph_Terms)):
            Graph = np.zeros(eval.shape[1:3])
            for k in range(eval.shape[1]):
                for l in range(eval.shape[2]):
                    if j == 9:
                        Graph[k, l] = eval[i, k, l, Graph_Terms[j] + 4]
                    else:
                        Graph[k, l] = eval[i, k, l, Graph_Terms[j] + 4]

            fig = plt.figure()
            fig.canvas.manager.set_window_title('Classes')
            plt.plot(Kfold, Graph[:, 0], color='#00FFFF', linewidth=3, marker='*',
                     markerfacecolor='#be03fd', markersize=16,
                     label="CO-ODRDANet")
            plt.plot(Kfold, Graph[:, 1], color='#9A32CD', linewidth=3, marker='*',
                     markerfacecolor='#fe02a2', markersize=16,
                     label="GOA-ODRDANet")
            plt.plot(Kfold, Graph[:, 2], color='#0804f9', linewidth=3, marker='*',
                     markerfacecolor='cyan', markersize=16,
                     label="WPA-ODRDANet")
            plt.plot(Kfold, Graph[:, 3], color='#02c14d', linewidth=3, marker='*',
                     markerfacecolor='#8B6969', markersize=16,
                     label="MA-ODRDANet")
            plt.plot(Kfold, Graph[:, 4], color='k', linewidth=3, marker='h', markerfacecolor='black',
                     markersize=12,
                     label="OMA-ODRDANet")
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.16),
                       ncol=3, fancybox=True, shadow=True)
            plt.xticks(Kfold, (
                'Bacterial \n Leaf Blight', 'Bacterial \n Leaf Streak', 'Bacterial \n Panicle Blight', 'Blast',
                'Brown \n Spot'))
            plt.ylabel(Terms[Graph_Terms[j]])
            # path = "./Results/%s_class_line.png" % (Terms[Graph_Terms[j]])
            # plt.savefig(path)
            plt.show()

            fig = plt.figure()
            ax = fig.add_axes([0.12, 0.12, 0.8, 0.8])
            fig.canvas.manager.set_window_title('Classes')
            X = np.arange(5)
            ax.bar(X + 0.00, Graph[:, 5], color='brown', width=0.10, label="Resnet")
            ax.bar(X + 0.10, Graph[:, 6], color='greenyellow', width=0.10, label="Shufflenet")
            ax.bar(X + 0.20, Graph[:, 7], color='yellow', width=0.10, label="Densenet")
            ax.bar(X + 0.30, Graph[:, 8], color='coral', width=0.10, label="DRDANet")
            ax.bar(X + 0.40, Graph[:, 4], color='k', width=0.10, label="OMA-ODRDANet")
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.13),
                       ncol=3, fancybox=True, shadow=True)
            plt.xticks(X + 0.20, (
                'Bacterial \n Leaf Blight', 'Bacterial \n Leaf Streak', 'Bacterial \n Panicle Blight', 'Blast',
                'Brown \n Spot'))
            plt.ylabel(Terms[Graph_Terms[j]])
            # path = "./Results/%s_class_bar.png" % (Terms[Graph_Terms[j]])
            # plt.savefig(path)
            plt.show()


def plot_Classes_Results():
    eval = np.load('Evaluate.npy', allow_pickle=True)
    Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1_score', 'MCC',
             'FOR', 'PT',
             'BA', 'FM', 'BM', 'MK', 'PLHR', 'lrminus', 'DOR', 'Prevalence', 'Threat Score']
    Graph_Terms = [0]
    Kfold = ['Dead_Heart', 'Downy_Mildew', 'Hispa', 'Normal', 'Tungro']

    for i in range(eval.shape[0]):
        for j in range(len(Graph_Terms)):
            Graph = np.zeros(eval.shape[1:3])
            for k in range(eval.shape[1]):
                for l in range(eval.shape[2]):
                    if j == 9:
                        Graph[k, l] = eval[i, k, l, Graph_Terms[j] + 4]
                    else:
                        Graph[k, l] = eval[i, k, l, Graph_Terms[j] + 4]

            fig = plt.figure()
            fig.canvas.manager.set_window_title('Classes')
            plt.plot(Kfold, Graph[:, 0], color='#00FFFF', linewidth=3, marker='*',
                     markerfacecolor='#be03fd', markersize=16,
                     label="CO-ODRDANet")
            plt.plot(Kfold, Graph[:, 1], color='#9A32CD', linewidth=3, marker='*',
                     markerfacecolor='#fe02a2', markersize=16,
                     label="GOA-ODRDANet")
            plt.plot(Kfold, Graph[:, 2], color='#0804f9', linewidth=3, marker='*',
                     markerfacecolor='cyan', markersize=16,
                     label="WPA-ODRDANet")
            plt.plot(Kfold, Graph[:, 3], color='#02c14d', linewidth=3, marker='*',
                     markerfacecolor='#8B6969', markersize=16,
                     label="MA-ODRDANet")
            plt.plot(Kfold, Graph[:, 4], color='k', linewidth=3, marker='h', markerfacecolor='black',
                     markersize=12,
                     label="OMA-ODRDANet")
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.16),
                       ncol=3, fancybox=True, shadow=True)
            plt.xticks(Kfold, ('Dead Heart', 'Downy Mildew', 'Hispa', 'Normal', 'Tungro'))
            plt.ylabel(Terms[Graph_Terms[j]])
            # path = "./Results/%s_classes_line.png" % (Terms[Graph_Terms[j]])
            # plt.savefig(path)
            plt.show()

            fig = plt.figure()
            ax = fig.add_axes([0.12, 0.12, 0.8, 0.8])
            fig.canvas.manager.set_window_title('Classes')
            X = np.arange(5)
            ax.bar(X + 0.00, Graph[:, 5], color='brown', width=0.10, label="Resnet")
            ax.bar(X + 0.10, Graph[:, 6], color='greenyellow', width=0.10, label="Shufflenet")
            ax.bar(X + 0.20, Graph[:, 7], color='yellow', width=0.10, label="Densenet")
            ax.bar(X + 0.30, Graph[:, 8], color='coral', width=0.10, label="DRDANet")
            ax.bar(X + 0.40, Graph[:, 4], color='k', width=0.10, label="OMA-ODRDANet")
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.13),
                       ncol=3, fancybox=True, shadow=True)
            plt.xticks(X + 0.20, ('Dead Heart', 'Downy Mildew', 'Hispa', 'Normal', 'Tungro'))
            plt.ylabel(Terms[Graph_Terms[j]])
            # path = "./Results/%s_classes_bar.png" % (Terms[Graph_Terms[j]])
            # plt.savefig(path)
            plt.show()


def Dice_Table():
    eval = np.load('Diec_Eval.npy', allow_pickle=True)
    Terms = ['Dice Coefficient']
    Classifier = ['TERMS', 'Unet', 'Unet++', 'ResUnet', 'DenseUnet', 'PROPOSED']
    Batch_size = [4, 8, 16, 32, 64]
    for k in range(eval.shape[0]):
        value1 = eval[k, :, 4:]

        Table = PrettyTable()
        Table.add_column(Classifier[0], Terms)
        for j in range(len(Classifier) - 1):
            Table.add_column(Classifier[j + 1], value1[j, :])
        print('--------------------------------------------------',k+1,  '-Batch_Size ',
              '-Classifier Comparison',
              '--------------------------------------------------')
        print(Table)


if __name__ == '__main__':
    plotConvResults()
    Kfold_plot_results()
    Plot_ROC_Curve()
    plot_seg_results()
    Epoch_Table()
    plot_Class_Results()
    plot_Classes_Results()
    Dice_Table()
