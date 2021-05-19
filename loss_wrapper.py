import torch
import numpy as np
from torch.autograd import Variable

from facial_landmark_extractor import FacialLandmarksExtractor

# Custom Banding Pattern Loss
class FacialLandmarkLoss(torch.autograd.Function):

    @staticmethod
    def forward(ctx, pred, target, FLE):

        img1_copy = target.clone().detach().numpy()
        img2_copy = pred.clone().detach().numpy()

        landmarks1 = FLE.extract(img1_copy)
        landmarks2 = FLE.extract(img2_copy)

        landmarks1, landmarks2 = FLE.landmarks_distance(landmarks1, landmarks2, normalise=True, no_calc=True)
        landmarks1 = Variable(torch.FloatTensor(landmarks1), requires_grad=True)
        landmarks2 = Variable(torch.FloatTensor(landmarks2), requires_grad=True)
        
        ctx.save_for_backward(landmarks1, landmarks2)

        return torch.sum(torch.sqrt((landmarks1 - landmarks2)**2))

    @staticmethod
    def backward(ctx, grad_output):
        landmarks1, landmarks2, = ctx.saved_tensors
        grad_input = 2 * (landmarks1 - landmarks2)

        return grad_input, None 
