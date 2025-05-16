import os
import cv2 as cv
from numpy import matlib
from sklearn.utils import shuffle
import random as rn
from AOA import AOA
from FCM import FCM
from FSA import FSA
from Glob_Vars import Glob_Vars
from Image_results import Image_Results
from Model_ANN import Model_ANN
from Model_AT_E_Unet2 import Model_AT_E_Unet2
from Model_CNN import Model_CNN
from Model_EfficientDet import Model_EfficientDet
from Model_M_EDM import Model_M_EDM
from Model_VGG16 import Model_VGG16
from Objective_Function import Objective_Seg
from PROPOSED import PROPOSED
from RUNet2 import RUNet2
from SAA import SAA
from SCO import SCO
from Unet import Unet
from YOLOv8 import YoloV8
from Plot_Results_Seg import *
from Plot_Results import *

No_of_dataset = 3

# Read Dataset 1
an = 0
if an == 1:
    Images = []
    GT = []
    Target = []
    path = './Dataset/Dataset 1/'
    out_dir = os.listdir(path)
    for i in range(len(out_dir)):
        files_path = path + '/' + out_dir[i]
        in_dir = os.listdir(files_path)
        for j in range(len(in_dir)):
            print(i, j)
            images = files_path + '/' + in_dir[j]
            if 'mask' not in in_dir[j]:
                img = cv.imread(images)
                img_re = cv.resize(img, [512, 512])
                Images.append(img_re)
                Target.append(out_dir[i])
            else:
                GT_img = cv.imread(images)
                GT_re = cv.resize(GT_img, [512, 512])
                GT.append(GT_re)
    Targ = np.asarray(Target)
    uni = np.unique(Targ)
    tar = np.zeros((Targ.shape[0], len(uni))).astype('int')
    for i in range(len(uni)):
        ind = np.where((Targ == uni[i]))
        tar[ind[0], i] = 1
    Images, GT, tar = shuffle(Images, GT, tar)
    np.save('Images_1.npy', Images)
    np.save('Ground_Truth_1.npy', GT)
    np.save('Targets_1.npy', tar)

# Dataset 2
an = 0
if an == 1:
    Images = []
    GT = []
    Target = []
    path = 'Dataset/Dataset 2/Training'
    out_dir = os.listdir(path)
    for i in range(len(out_dir)):
        files = path + '/' + out_dir[i]
        in_dir = os.listdir(files)
        for j in range(len(in_dir)):
            print(i, j)
            Image = files + '/' + in_dir[j]
            img = cv.imread(Image)
            img_re = cv.resize(img, [512, 512])
            Images.append(img_re)
            Target.append(out_dir[i])
    Targ = np.asarray(Target)
    uni = np.unique(Targ)
    tar = np.zeros((Targ.shape[0], len(uni))).astype('int')
    for i in range(len(uni)):
        ind = np.where((Targ == uni[i]))
        tar[ind[0], i] = 1
    np.save('Images_2.npy', Images)
    np.save('Targets_2.npy', tar)

# Dataset 3
an = 0
if an == 1:
    Images = []
    GT = []
    Target = []
    path = 'Dataset/Dataset 3'
    out_dir = os.listdir(path)
    for i in range(len(out_dir)):
        files = path + '/' + out_dir[i]
        in_dir = os.listdir(files)
        for j in range(len(in_dir)):
            print(i, j)
            Image = files + '/' + in_dir[j]
            img = cv.imread(Image)
            img_re = cv.resize(img, [512, 512])
            Images.append(img_re)
            Target.append(out_dir[i])
    Targ = np.asarray(Target)
    uni = np.unique(Targ)
    tar = np.zeros((Targ.shape[0], len(uni))).astype('int')
    for i in range(len(uni)):
        ind = np.where((Targ == uni[i]))
        tar[ind[0], i] = 1
    np.save('Images_3.npy', Images)
    np.save('Targets_3.npy', tar)

# GroundTruth Dataset 2 & 3
an = 0
if an == 1:
    for i in range(No_of_dataset - 1):
        GT = []
        Images = np.load('Images_' + str(i + 2) + '.npy', allow_pickle=True)
        for k in range(len(Images)):
            print('Image', k)
            img = Images[k]
            imag = cv.resize(img, [512, 512])
            image = cv.cvtColor(imag, cv.COLOR_BGR2GRAY)
            cluster = FCM(image, image_bit=8, n_clusters=8, m=10, epsilon=0.8, max_iter=30)
            cluster.form_clusters()
            result = cluster.result.astype('uint8') * 30
            values, counts = np.unique(result, return_counts=True)
            index = np.argsort(counts)[::-1][2]
            result[result != values[index]] = 0
            analysis = cv.connectedComponentsWithStats(result, 4, cv.CV_32S)
            (totalLabels, Img, values, centroid) = analysis
            uniq, counts = np.unique(Img, return_counts=True)
            zeroIndex = np.where(uniq == 0)[0][0]
            uniq = np.delete(uniq, zeroIndex)
            counts = np.delete(counts, zeroIndex)
            sortIndex = np.argsort(counts)[::-1]
            uniq = uniq[sortIndex]
            counts = counts[sortIndex]
            Img = Img.astype('uint8')
            remArray = []
            for j in range(len(counts)):
                if counts[j] < 100 or counts[j] > 750:
                    Img[Img == uniq[j]] = 0
                    remArray.append(j)
            if not remArray:
                pass
            else:
                remArray = np.array(remArray)
                uniq = np.delete(uniq, remArray)
                counts = np.delete(counts, remArray)
                Img[Img != 0] = 255
                Img = Img.astype('uint8')
                kernel = np.ones((3, 3), np.uint8)
                opening = cv.morphologyEx(Img, cv.MORPH_OPEN, kernel, iterations=1)
                kernel = np.ones((3, 3), np.uint8)
                closing = cv.morphologyEx(opening, cv.MORPH_CLOSE, kernel, iterations=1).astype('uint8')
                GT.append(closing)
        np.save('Ground_Truth_' + str(i + 1) + '.npy', GT)

# Detection
an = 0
if an == 1:
    EV = []
    for k in range(No_of_dataset):
        Images = np.load('Images_' + str(k + 1) + '.npy', allow_pickle=True)
        Target = np.load('Targets_' + str(k + 1) + '.npy', allow_pickle=True)
        Eval = []
        Hidden_Neuron_Count = [50, 100, 150, 200, 250]
        for i in range(len(Hidden_Neuron_Count)):
            learnperc = round(Images.shape[0] * 0.75)
            Train_Data = Images[:learnperc, :]
            Train_Target = Target[:learnperc, :]
            Test_Data = Images[learnperc:, :]
            Test_Target = Target[learnperc:, :]
            EVAL = np.zeros((5, 25))
            EVAL[0, :] = Model_CNN(Train_Data, Train_Target, Test_Data, Test_Target)
            EVAL[1, :] = Model_ANN(Train_Data, Train_Target, Test_Data, Test_Target)
            EVAL[2, :] = Model_VGG16(Train_Data, Train_Target, Test_Data, Test_Target)
            EVAL[3, :] = Model_EfficientDet(Train_Data, Train_Target)
            EVAL[4, :], Image4 = Model_M_EDM(Train_Data, Train_Target)
            Eval.append(EVAL)
        np.save('Detection_Images_' + str(k + 1) + '.npy', Image4)
        EV.append(Eval)

    np.save('Eval_Hidden.npy', EV)

# Optimization for Segmentation
an = 0
if an == 1:
    sol = []
    fitness = []
    for k in range(No_of_dataset):
        Images = np.load('Detection_Images_' + str(k + 1) + '.npy', allow_pickle=True)
        Target = np.load('Targets_' + str(k + 1) + '.npy', allow_pickle=True)
        Glob_Vars.Data = Images
        Glob_Vars.Target = Target
        fname = Objective_Seg
        Npop = 10
        Chlen = 3
        max_iter = 50
        xmin = matlib.repmat(([5, 0.01, 100]), Npop, 1)
        xmax = matlib.repmat(([255, 0.99, 500]), Npop, 1)
        initsol = np.zeros(xmin.shape)
        for i in range(xmin.shape[0]):
            for j in range(xmin.shape[1]):
                initsol[i, j] = rn.uniform(xmin[i, j], xmax[i, j])

        print('FSA....')
        [bestfit1, fitness1, bestsol1, Time1] = FSA(initsol, fname, xmin, xmax, max_iter)

        print('AOA....')
        [bestfit2, fitness2, bestsol2, Time2] = AOA(initsol, fname, xmin, xmax, max_iter)

        print('SCO....')
        [bestfit3, fitness3, bestsol3, Time3] = SCO(initsol, fname, xmin, xmax, max_iter)

        print('SAA....')
        [bestfit4, fitness4, bestsol4, Time4] = SAA(initsol, fname, xmin, xmax, max_iter)

        print('PROPOSED....')
        [bestfit5, fitness5, bestsol5, Time5] = PROPOSED(initsol, fname, xmin, xmax, max_iter)

        sol.append([bestsol1, bestsol2, bestsol3, bestsol4, bestsol5])
        fitness.append([fitness1.ravel(), fitness2.ravel(), fitness3.ravel(), fitness4.ravel(), fitness5.ravel()])

    np.save('Bestsol.npy', sol)
    np.save('Fitness.npy', fitness)


# Segmentation
an = 0
if an == 1:
    Eval_all = []
    for a in range(No_of_dataset):
        Images = np.load('Detection_Images_' + str(a + 1) + '.npy', allow_pickle=True)
        Target = np.load('Ground_Truth_' + str(a + 1) + '.npy', allow_pickle=True)
        sol = np.load('Bestsol.npy', allow_pickle=True)[a]
        Eval = np.zeros((10, 3))
        for i in range(5):
            Eval[i, :], Image5 = Model_AT_E_Unet2(Images, Target, sol)
        Eval[5, :], Image1 = RUNet2(Images, Target)
        Eval[6, :], Image2 = Unet(Images, Target)
        Eval[7, :], Image3 = YoloV8(Images, Target)
        Eval[8, :], Image4 = Model_AT_E_Unet2(Images, Target)
        Eval[9, :] = Eval[4, :]
        np.save('Segmented_Images_' + str(a + 1) + '.npy', Image5)
        Eval_all.append(Eval)
    np.save('Eval_all_seg.npy', Eval_all)

Image_Results()
plot_results_seg_table()
Plot_Results_seg()
plot_results_conv()
ROC_curve()
Plot_Results()
Plot_table()