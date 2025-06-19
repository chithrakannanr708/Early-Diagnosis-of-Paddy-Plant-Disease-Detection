import numpy as np
from Evaluation import evaluation
from Global_Vars import Global_Vars
from Model_Res_DenseNet import Model_Res_DenseNet


def objfun_cls(Soln):
    data = Global_Vars.Data
    Tar = Global_Vars.Target
    Fitn = np.zeros(Soln.shape[0])
    dimension = len(Soln.shape)
    if dimension == 2:
        learnper = round(data.shape[0] * 0.75)
        for i in range(Soln.shape[0]):
            sol = np.round(Soln[i, :]).astype(np.int16)
            Train_Data = data[:learnper, :]
            Train_Target = Tar[:learnper, :]
            Test_Data = data[learnper:, :]
            Test_Target = Tar[learnper:, :]
            Eval, pred = Model_Res_DenseNet(Train_Data, Train_Target, Test_Data, Test_Target, sol)
            Eval = evaluation(pred, Test_Target)
            Fitn[i] = (1 / Eval[4]) + Eval[8]
        return Fitn
    else:
        learnper = round(data.shape[0] * 0.75)
        sol = np.round(Soln).astype(np.int16)
        Train_Data = data[:learnper, :]
        Train_Target = Tar[:learnper, :]
        Test_Data = data[learnper:, :]
        Test_Target = Tar[learnper:, :]
        Eval, pred = Model_Res_DenseNet(Train_Data, Train_Target, Test_Data, Test_Target, sol)
        Eval = evaluation(pred, Test_Target)
        Fitn = (1 / Eval[4]) + Eval[8]
        return Fitn
