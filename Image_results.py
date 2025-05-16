import numpy as np
import cv2 as cv
import matplotlib

matplotlib.use('TkAgg')
import matplotlib.pyplot as plt

no_of_dataset = 3


def Image_Results():
    for i in range(no_of_dataset):
        Orig = np.load('Ori_' + str(i + 1) + '.npy', allow_pickle=True)
        G_T = np.load('GT_' + str(i + 1) + '.npy', allow_pickle=True)
        det = np.load('Detection_Images_' + str(i + 1) + '.npy', allow_pickle=True)
        Image1 = np.load('RUNet2_' + str(i + 1) + '.npy', allow_pickle=True)
        Image2 = np.load('Unet_' + str(i + 1) + '.npy', allow_pickle=True)
        Image3 = np.load('YOLOv8_' + str(i + 1) + '.npy', allow_pickle=True)
        Image4 = np.load('E_Unet2_' + str(i + 1) + '.npy', allow_pickle=True)
        segment = np.load('Proposed_' + str(i + 1) + '.npy', allow_pickle=True)
        for j in range(5):
            original = Orig[j]
            GT = G_T[j]
            dt = det[j]
            image1 = Image1[j]
            image2 = Image2[j]
            image3 = Image3[j]
            image4 = Image4[j]
            seg = segment[j]
            Output1 = np.zeros((image1.shape)).astype('uint8')
            ind1 = np.where(image1 > 0)
            Output1[ind1] = 255

            Output2 = np.zeros((image2.shape)).astype('uint8')
            ind2 = np.where(image2 > 0)
            Output2[ind2] = 255

            Output3 = np.zeros((image3.shape)).astype('uint8')
            ind3 = np.where(image3 > 0)
            Output3[ind3] = 255

            Output4 = np.zeros((image4.shape)).astype('uint8')
            ind4 = np.where(image4 > 0)
            Output4[ind4] = 255

            Output5 = np.zeros((seg.shape)).astype('uint8')
            ind5 = np.where(seg > 0)
            Output5[ind5] = 255

            Output6 = np.zeros((GT.shape)).astype('uint8')
            ind6 = np.where(GT > 0)
            Output6[ind6] = 255

            plt.suptitle(" Dataset %d - Image %d" % ((i + 1), (j + 1)), fontsize=20)
            plt.subplot(2, 4, 1)
            plt.title('Original')
            plt.imshow(original)

            plt.subplot(2, 4, 2)
            plt.title('Detection')
            plt.imshow(dt)

            plt.subplot(2, 4, 3)
            plt.title('RUNet2')
            plt.imshow(Output1)

            plt.subplot(2, 4, 4)
            plt.title('Unet')
            plt.imshow(Output2)

            plt.subplot(2, 4, 5)
            plt.title('YOLOv8')
            plt.imshow(Output3)

            plt.subplot(2, 4, 6)
            plt.title('EUnet2')
            plt.imshow(Output4)

            plt.subplot(2, 4, 7)
            plt.title('PROPOSED')
            plt.imshow(Output5)
            plt.show(block=False)
            plt.pause(2)
            plt.close()

            cv.imwrite('./Results/Segmented Images/Dataset ' + str(i + 1) + '/Original-' + str(j + 1) + '.png',
                       original)
            cv.imwrite('./Results/Segmented Images/Dataset ' + str(i + 1) + '/GT-' + str(j + 1) + '.png', GT)
            cv.imwrite('./Results/Segmented Images/Dataset ' + str(i + 1) + '/Detection-' + str(j + 1) + '.png', dt)
            cv.imwrite('./Results/Segmented Images/Dataset ' + str(i + 1) + '/RUNet2-' + str(j + 1) + '.png', image1)
            cv.imwrite('./Results/Segmented Images/Dataset ' + str(i + 1) + '/Unet-' + str(j + 1) + '.png',
                       image2)
            cv.imwrite('./Results/Segmented Images/Dataset ' + str(i + 1) + '/YOLOv8-' + str(j + 1) + '.png',
                       image3)
            cv.imwrite('./Results/Segmented Images/Dataset ' + str(i + 1) + '/EUnet2-' + str(j + 1) + '.png',
                       image4)
            cv.imwrite('./Results/Segmented Images/Dataset ' + str(i + 1) + '/PROPOSED-' + str(j + 1) + '.png', seg)


if __name__ == '__main__':
    Image_Results()
