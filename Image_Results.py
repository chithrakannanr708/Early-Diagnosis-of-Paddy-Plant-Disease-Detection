import matplotlib.pyplot as plt
import numpy as np
import cv2 as cv


def Sample_Images():
    Orig = np.load('Image.npy', allow_pickle=True)
    ind = [388, 398, 400, 405, 410, 450]
    fig, ax = plt.subplots(2, 3)
    plt.suptitle("Sample Images from Dataset")
    plt.subplot(2, 3, 1)
    plt.title('Image-1')
    plt.imshow(Orig[ind[0]])
    plt.subplot(2, 3, 2)
    plt.title('Image-2')
    plt.imshow(Orig[ind[1]])
    plt.subplot(2, 3, 3)
    plt.title('Image-3')
    plt.imshow(Orig[ind[2]])
    plt.subplot(2, 3, 4)
    plt.title('Image-4')
    plt.imshow(Orig[ind[3]])
    plt.subplot(2, 3, 5)
    plt.title('Image-5')
    plt.imshow(Orig[ind[4]])
    plt.subplot(2, 3, 6)
    plt.title('Image-6')
    plt.imshow(Orig[ind[5]])
    plt.show()


def Image_Results():
    I = [[0, 2, 6, 4, 11]]
    for n in range(1):
        Images = np.load('Image.npy', allow_pickle=True)
        GT = np.load('Seg_Image.npy', allow_pickle=True)
        for i in range(len(I[n])):
            plt.subplot(1, 2, 1)
            plt.title('Original')
            plt.imshow(Images[I[n][i]])
            plt.subplot(1, 2, 2)
            plt.title('Segmentation')
            plt.imshow(GT[I[n][i]])
            plt.tight_layout()
            # cv.imwrite('./Results/Image_Results/' + 'orig-' + str(i + 1) + '.png', Images[I[0][i]])
            # cv.imwrite('./Results/Image_Results/' + 'Seg-' + str(i + 1) + '.png', GT[I[0][i]])
            plt.show()


if __name__ == '__main__':
    Image_Results()
    Sample_Images()
