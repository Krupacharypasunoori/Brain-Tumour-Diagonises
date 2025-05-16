import numpy as np
from matplotlib import pyplot as plt
from prettytable import PrettyTable
import seaborn as sns

No_of_dataset = 3



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
            plt.show()

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
            plt.show()


if __name__ == '__main__':
    plot_results_seg_table()
    Plot_Results_seg()
