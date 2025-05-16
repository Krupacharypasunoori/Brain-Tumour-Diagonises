from itertools import cycle
import numpy as np
from matplotlib import pyplot as plt
from prettytable import PrettyTable
from sklearn.metrics import roc_curve

No_of_dataset = 3


def ROC_curve():
    lw = 2
    cls = ['CNN', 'ANN', 'VGG16', 'EfficientDet', 'MEDet']
    for k in range(No_of_dataset):
        # Create a figure and axis
        fig, ax = plt.subplots()
        fig.canvas.manager.set_window_title('Roc Curve')
        Actual = np.load('Targets_' + str(k + 1) + '.npy', allow_pickle=True).astype('int')
        per = round(Actual.shape[0] * 0.75)
        Actual = Actual[per:, :]
        colors = cycle(["#fe2f4a", "#8b88f8", "#fc824a", "lime", "black"])
        for i, color in zip(range(len(cls)), colors):
            Predicted = np.load('Y_Score_' + str(k + 1) + '.npy', allow_pickle=True)[i]
            false_positive_rate1, true_positive_rate1, threshold1 = roc_curve(Actual.ravel(), Predicted.ravel())
            plt.plot(
                false_positive_rate1,
                true_positive_rate1,
                color=color,
                lw=lw,
                label=cls[i],
            )
        plt.plot([0, 1], [0, 1], "k--", lw=lw)
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel("False Positive Rate")
        plt.ylabel("True Positive Rate")
        plt.title("ROC Curve")
        plt.legend(loc="lower right")
        path = "./Results/ROC_%s.png" % (str(k + 1))
        plt.savefig(path)
        plt.show(block=False)
        plt.pause(2)
        plt.close()


def plot_results_conv():
    for s in range(No_of_dataset):
        conv = np.load('Fitness.npy', allow_pickle=True)[s]
        ind = np.argsort(conv[:, conv.shape[1] - 1])
        x = conv[ind[0], :].copy()
        y = conv[4, :].copy()
        conv[4, :] = x
        conv[ind[0], :] = y

        Statistics = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD']
        Algorithm = ['FSA-ATEUnet2+', 'AOA-ATEUnet2+', 'SCO-ATEUnet2+', 'SAA-ATEUnet2+', 'RASAA-ATEUnet2+']
        color = ['green', 'cyan', 'deeppink', 'green', 'k']
        markerfacecolor = ['red', 'green', 'cyan', 'y', 'black']
        Value = np.zeros((conv.shape[0], 5))
        for j in range(conv.shape[0]):
            Value[j, 0] = np.min(conv[j, :])
            Value[j, 1] = np.max(conv[j, :])
            Value[j, 2] = np.mean(conv[j, :])
            Value[j, 3] = np.median(conv[j, :])
            Value[j, 4] = np.std(conv[j, :])

        Table = PrettyTable()
        Table.add_column("Statistical", Statistics)
        for j in range(len(Algorithm)):
            Table.add_column(Algorithm[j], Value[j, :])
        print('----------------------------------------Dataset ' + str(s + 1) +
              ' Statistical Analysis -------------------------------')
        print(Table)

        fig = plt.figure()
        fig.canvas.manager.set_window_title('Convergence')
        iteration = np.arange(conv.shape[1])
        for m in range(conv.shape[0]):
            plt.plot(iteration, conv[m, :], color=color[m], linewidth=3, marker='*',
                     markerfacecolor=markerfacecolor[m], markersize=12, label=Algorithm[m])
        plt.xlabel('Iteration')
        plt.ylabel('Cost Function')
        plt.legend(loc=1)
        path = "./Results/Conv_%s.png" % (str(s + 1))
        plt.savefig(path)
        plt.show(block=False)
        plt.pause(2)
        plt.close()


def plot_results_seg_table():
    Eval_all = np.load('Eval_all_seg.npy', allow_pickle=True)
    Terms = ['Dice Coefficient', 'Jaccard', 'Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR',
             'NPV',
             'FDR', 'F1-Score', 'MCC']
    Dataset = ['Dataset 1', 'Dataset 2', 'Dataset 3']

    Statistics = ['BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD']

    Full = ['TERMS', 'RU-Net2+', 'Unet', 'YOLOv8 ', 'TEUnet2+', 'RASAA-ATEUnet2+']
    Alg = ['TERMS', 'FSA-ATEUnet2+', 'AOA-ATEUnet2+', 'SCO-ATEUnet2+', 'SAA-ATEUnet2+', 'RASAA-ATEUnet2+']

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

            Table = PrettyTable()
            Table.add_column(Alg[0], Statistics)
            for k in range(5):
                Table.add_column(Alg[k + 1], stats[i, k, :])
            print('-------------------------------------------------- ', Terms[i - 4],
                  'Algorthm  for Segmentation of dataset - ', n + 1,
                  '--------------------------------------------------')
            print(Table)

            Table = PrettyTable()
            Table.add_column(Full[0], Statistics)
            Table.add_column(Full[1], stats[i, 5, :])
            Table.add_column(Full[2], stats[i, 6, :])
            Table.add_column(Full[3], stats[i, 7, :])
            Table.add_column(Full[4], stats[i, 8, :])
            Table.add_column(Full[5], stats[i, 4, :])
            print('-------------------------------------------------- ', Terms[i - 4],
                  'Comparison for Segmentation of dataset', n + 1, '--------------------------------------------------')
            print(Table)


def Plot_Results_seg():
    for k in range(No_of_dataset):
        Terms = ['Dice Coefficient', 'IOU', 'Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR',
                 'NPV', 'FDR', 'F1-Score', 'MCC']
        Graph_Term = [0, 1, 2]  # Metrics to plot
        Algorithm = ['FSA-ATEUnet2+', 'AOA-ATEUnet2+', 'SCO-ATEUnet2+', 'SAA-ATEUnet2+', 'RASAA-ATEUnet2+']
        Classifier = ['RU-Net2+', 'Unet', 'YOLOv8 ', 'TEUnet2+', 'RASAA-ATEUnet2+']

        Eval = np.load('Eval_Seg_Kfold.npy', allow_pickle=True)[k]
        Kfold = [1, 2, 3, 4, 5]  # Updated Kfold values
        colors_alg = ['#8C000F', '#0343DF', '#FC5A50', '#15B01A', '#800080']  # Algorithm colors
        colors_met = ['orange', 'teal', 'violet', 'olive', 'k']

        for m in range(len(Graph_Term)):
            Graph = Eval[:, :5, Graph_Term[m] + 4]

            # Start plotting
            plt.figure(figsize=(9, 6))
            for s in range(Graph.shape[1]):
                plt.plot(Kfold, Graph[:, s], label=Algorithm[s], color=colors_alg[s], linewidth=2, marker='o')

            # Add labels, title, and legend
            plt.xlabel('KFOLD', fontsize=12)
            plt.ylabel(Terms[Graph_Term[m]], fontsize=12)
            # Position legend at the top of the graph
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), fontsize=10, ncol=len(Classifier))

            # Alternate shading for better readability
            for i in range(len(Kfold)):
                if i % 2 == 0:
                    plt.axvspan(Kfold[i] - 0.5, Kfold[i] + 0.5, color='lightgray', alpha=0.2)

            # Remove spines for a cleaner look
            for spine in plt.gca().spines.values():
                spine.set_visible(False)

            # Save and show the plot
            Path = f"./Results/Dataset_{k + 1}_{Terms[Graph_Term[m]]}_Seg_Algorthim.png"
            plt.savefig(Path, dpi=300, bbox_inches='tight')
            plt.show(block=False)
            plt.pause(2)
            plt.close()

            Graph = Eval[:, 5:, Graph_Term[m] + 4]

            # Start plotting
            plt.figure(figsize=(9, 6))
            for s in range(Graph.shape[1]):
                plt.plot(Kfold, Graph[:, s], label=Classifier[s], color=colors_met[s], linewidth=2, marker='o')

            # Add labels, title, and legend
            plt.xlabel('KFOLD', fontsize=12)
            plt.ylabel(Terms[Graph_Term[m]], fontsize=12)
            # Position legend at the top of the graph
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), fontsize=10, ncol=len(Classifier))

            # Alternate shading for better readability
            for i in range(len(Kfold)):
                if i % 2 == 0:
                    plt.axvspan(Kfold[i] - 0.5, Kfold[i] + 0.5, color='lightgray', alpha=0.2)

            # Remove spines for a cleaner look
            for spine in plt.gca().spines.values():
                spine.set_visible(False)

            # Save and show the plot
            Path = f"./Results/Dataset_{k + 1}_{Terms[Graph_Term[m]]}_Seg_Method.png"
            plt.savefig(Path, dpi=300, bbox_inches='tight')
            plt.show(block=False)
            plt.pause(2)
            plt.close()


def plot_results_Detect():
    Eval_all = np.load('Eval_all_Det.npy', allow_pickle=True)
    Terms = ['Dice Coefficient', 'Jaccard', 'Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR',
             'NPV',
             'FDR', 'F1-Score', 'MCC']
    Dataset = ['Dataset 1', 'Dataset 2', 'Dataset 3']

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

            X = np.arange(stats.shape[2])

            fig = plt.figure()
            ax = fig.add_axes([0.1, 0.1, 0.8, 0.8])
            ax.bar(X + 0.20, stats[i, 7, :], color='r', hatch='*', edgecolor='k', width=0.10, label="YoloV3")
            ax.bar(X + 0.30, stats[i, 8, :], color='c', hatch='*', edgecolor='k', width=0.10, label="YoloV5")
            ax.bar(X + 0.40, stats[i, 4, :], color='k', hatch='\\', edgecolor='w', width=0.10,
                   label="M-EDM")
            plt.xticks(X + 0.20, ('BEST', 'WORST', 'MEAN', 'MEDIAN', 'STD'))
            plt.xlabel('Statisticsal Analysis')
            plt.ylabel(Terms[i - 4])
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15),
                       ncol=3, fancybox=True, shadow=True)
            path1 = "./Results/%s_Detected_%s.png" % (Dataset[n], Terms[i - 4])
            plt.savefig(path1)
            plt.show(block=False)
            plt.pause(2)
            plt.close()


if __name__ == '__main__':
    plot_results_Detect()
    plot_results_conv()
    ROC_curve()
    Plot_Results_seg()
    plot_results_seg_table()
