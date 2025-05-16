import numpy as np
from Evaluation_All import seg_evaluation
from Glob_Vars import Glob_Vars
from Model_AT_E_Unet2 import Model_AT_E_Unet2


def Objective_Seg(Soln):
    Images = Glob_Vars.Images
    Target = Glob_Vars.Target
    Fitn = np.zeros(Soln.shape[0])
    dimension = len(Soln.shape)
    if dimension == 2:
        for i in range(Soln.shape[0]):
            sol = np.round(Soln[i, :]).astype(np.int16)
            pred = Model_AT_E_Unet2(Images, Target, sol)
            Eval = seg_evaluation(pred, Target)
            Fitn[i] = 1 / Eval[5]  # IoU
        return Fitn
    else:
        sol = np.round(Soln).astype(np.int16)
        pred = Model_AT_E_Unet2(Images, Target, sol)
        Eval = seg_evaluation(pred, Target)
        Fitn = 1 / Eval[5]  # IoU
        return Fitn
