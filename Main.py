import os
import pandas as pd
from numpy import matlib
import random as rn
from CO import CO
from FasterRCNN import Model_RCNN
from GOA import GOA
from Global_Vars import Global_Vars
from MA import MA
from Model_DenseNet import Model_DenseNet
from Model_Res_DenseNet import Model_Res_DenseNet
from Model_Resnet import Model_Resnet
from Model_Sufflenet import Model_shufflenet
from Plot_Results import *
from Proposed import Proposed
from WPA import WPA
from objfun_feat import objfun_cls

# Read Dataset
an = 0
if an == 1:
    Images = []
    Tar = []
    path = './Dataset'
    out_dir = os.listdir(path)
    for i in range(len(out_dir)):
        in_dir = path + '/' + out_dir[i]
        folder_name = os.listdir(in_dir)
        for j in range(len(folder_name)):
            Image_Name = in_dir + '/' + folder_name[j]
            Image = cv.imread(Image_Name)
            Resize_img = cv.resize(Image, (256, 256))
            Images.append(Resize_img)
            Tar.append(i)
    Img = np.resize(Images, (10407, 256, 256, 3))
    df = pd.DataFrame(Tar)
    new_df = df.fillna(0)
    uniq = df[0].unique()
    Target = np.asarray(df[0])
    target = np.zeros((Target.shape[0], len(uniq)))  # create within rage zero values
    for uni in range(len(uniq)):
        index = np.where(Target == uniq[uni])
        target[index[0], uni] = 1
    # np.save('Image.npy', Img)
    np.save('Target.npy', target)

# Segmentation
an = 0
if an == 1:
    Seg_Image = []
    Image = np.load('Image.npy', allow_pickle=True)
    Img = Model_RCNN()
    Seg_Image.append(Img)
    np.save('Seg_Image.npy', Seg_Image)

# Optimization for Classification
an = 0
if an == 1:
    Data = np.load('Seg_Image.npy', allow_pickle=True)
    Target = np.load('Target.npy', allow_pickle=True)
    Global_Vars.Data = Data
    Global_Vars.Target = Target
    Npop = 10
    Chlen = 3  # Hidden Neuron, Epochs, Step epoch
    xmin = matlib.repmat([5, 5, 50], Npop, 1)
    xmax = matlib.repmat([255, 50, 250], Npop, 1)
    fname = objfun_cls
    initsol = np.zeros((Npop, Chlen))
    for p1 in range(initsol.shape[0]):
        for p2 in range(initsol.shape[1]):
            initsol[p1, p2] = rn.uniform(xmin[p1, p2], xmax[p1, p2])
    Max_iter = 50

    print("CO...")
    [bestfit1, fitness1, bestsol1, time] = CO(initsol, fname, xmin, xmax, Max_iter)  # CO

    print("GOA...")
    [bestfit2, fitness2, bestsol2, time1] = GOA(initsol, fname, xmin, xmax, Max_iter)  # GOA

    print("WPA...")
    [bestfit3, fitness3, bestsol3, time2] = WPA(initsol, fname, xmin, xmax, Max_iter)  # WPA

    print("MA...")
    [bestfit4, fitness4, bestsol4, time3] = MA(initsol, fname, xmin, xmax, Max_iter)  # MA

    print("Proposed...")
    [bestfit5, fitness5, bestsol5, time4] = Proposed(initsol, fname, xmin, xmax, Max_iter)  # Proposed

    BestSol = [bestsol1, bestsol2, bestsol3, bestsol4, bestsol5]
    np.save('BestSol_CLS.npy', BestSol)

# Classification
an = 0
if an == 1:
    Feature = np.load('Seg_Image.npy', allow_pickle=True)  # loading step
    Target = np.load('Target.npy', allow_pickle=True)  # loading step
    BestSol = np.load('BestSol_CLS.npy', allow_pickle=True)  # loading step
    K = 5
    Per = 1 / 5
    Perc = round(Feature.shape[0] * Per)
    eval = []
    for i in range(K):
        Eval = np.zeros((5, 25))
        for j in range(5):
            Feat = Feature
            sol = np.round(BestSol[j, :]).astype(np.int16)
            Test_Data = Feat[i * Perc: ((i + 1) * Perc), :]
            Test_Target = Target[i * Perc: ((i + 1) * Perc), :]
            test_index = np.arange(i * Perc, ((i + 1) * Perc))
            total_index = np.arange(Feat.shape[0])
            train_index = np.setdiff1d(total_index, test_index)
            Train_Data = Feat[train_index, :]
            Train_Target = Target[train_index, :]
            Eval[j, :], pred = Model_Res_DenseNet(Train_Data, Train_Target, Test_Data, Test_Target,
                                              sol)  # Residual + DenseNet With optimization
        Eval[5, :], pred1 = Model_Resnet(Train_Data, Train_Target, Test_Data, Test_Target)  # Model Resnet
        Eval[6, :], pred2 = Model_shufflenet(Train_Data, Train_Target, Test_Data, Test_Target)  # Model Shufflenet
        Eval[7, :], pred3 = Model_DenseNet(Train_Data, Train_Target, Test_Data, Test_Target)  # Model DenseNet
        Eval[8, :], pred4 = Model_Res_DenseNet(Train_Data, Train_Target, Test_Data,
                                               Test_Target)  # Residual + DenseNet Without optimization
        Eval[9, :] = Eval[4, :]
        eval.append(Eval)
    np.save('Eval_all_KFold.npy', eval)  # Save Eval all

plotConvResults()
Kfold_plot_results()
Plot_ROC_Curve()
plot_seg_results()
Epoch_Table()
Sample_Images()
