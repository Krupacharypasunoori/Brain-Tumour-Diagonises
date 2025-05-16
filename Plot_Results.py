from itertools import cycle
import numpy as np
from matplotlib import pyplot as plt
from prettytable import PrettyTable
from sklearn.metrics import roc_curve

No_of_dataset = 3


def Plot_Results():
    for k in range(No_of_dataset):
        Terms = ['Accuracy', 'Sensitivity', 'Specificity', 'Precision', 'FPR', 'FNR', 'NPV', 'FDR', 'F1_score', 'MCC',
                 'FOR', 'pt', 'ba', 'fm', 'bm', 'mk', 'PLHR', 'lrminus', 'dor', 'prevalence', 'TS']
        Graph_Term = [0, 1, 2, 3, 4, 5, 6, 7, 8]  # Metrics to plot
        Classifier = [ 'CNN', 'ANN', 'VGG16', 'EfficientDet', 'MEDet']

        Eval = np.load('Eval_Hidden.npy', allow_pickle=True)[k]
        Hidden = [50, 100, 150, 200, 250]  # Hidden neuron counts
        colors = ['b', 'lime', 'deeppink', '#ef4026', 'k']  # Classifier colors

        for m in range(len(Graph_Term)):
            Graph = Eval[:, :, Graph_Term[m] + 4]

            # Start plotting
            plt.figure(figsize=(9, 6))
            for s in range(Graph.shape[1]):
                plt.plot(Hidden, Graph[:, s], label=Classifier[s], color=colors[s], linewidth=2)

            # Add labels, title, and legend
            plt.xlabel('Hidden Neuron Count', fontsize=12)
            plt.ylabel(Terms[Graph_Term[m]], fontsize=12)
            # Position legend at the top of the graph
            plt.legend(loc='upper center', bbox_to_anchor=(0.5, 1.15), fontsize=10, ncol=len(Classifier))

            # Alternate shading for better readability
            for i in range(len(Hidden)):
                if i % 2 == 0:
                    plt.axvspan(Hidden[i] - 25, Hidden[i] + 25, color='lightgray', alpha=0.2)

            # Remove spines for a cleaner look
            for spine in plt.gca().spines.values():
                spine.set_visible(False)

            # Save and show the plot
            Path = f"./Results/Dataset_{k + 1}_{Terms[Graph_Term[m]]}_Method.png"
            plt.savefig(Path, dpi=300, bbox_inches='tight')
            plt.show(block=False)
            plt.pause(2)
            plt.close()


def Plot_table():
    for s in range(No_of_dataset):
        Eval = np.load('Eval_Act.npy', allow_pickle=True)[s]
        Graph_Term = np.array([0]).astype(int)
        Comparison = ['Classifier']
        Networks = [['Activation Function', 'CNN', 'ANN', 'VGG16', 'EfficientDet', 'MEDet']]
        variation = ['Linear', 'ReLU', 'Tanh', 'Softmax', 'Sigmoid']
        value = Eval[:, :, 4:]
        for m in range(len(Comparison)):
            Table = PrettyTable()
            Table.add_column(Networks[m][0], variation)
            for j in range(len(Networks[m]) - 1):
                if m == 1:  # for Network
                    k = len(Networks[m]) + j - 1
                else:
                    k = j
                Table.add_column(Networks[m][j + 1],
                                 [item for sublist in value[:, k, Graph_Term] for item in sublist])
            print(
                '----------------------------------Dataset - ' + str(s + 1) + ' - Activation Function  -' +
                Comparison[m] + ' Comparison - Accuracy  -------------------------------')
            print(Table)


def ROC_curve():
    lw = 2
    cls = [ 'CNN', 'ANN', 'VGG16', 'EfficientDet', 'MEDet']
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


if __name__ == '__main__':
    # plot_results_conv()
    # Plot_Results()
    Plot_table()
    # ROC_curve()
