import math

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch import nn
import numpy as np
class my_loss_func(nn.Module):
    def __init__(self, n_classes, alph=1.0, tao=1.0):
        super(my_loss_func, self).__init__()
        self.classes = n_classes
        self.alph = alph
        self.tao = tao
    def forward(self, score, label):

        len = label.size(0)
        total_loss = 0
        for t in range(len):
            index = (np.argmax(label[t].cpu(), axis=0)).numpy()
            score_y = score[t][index]
            loss = 0
            for i in range (self.classes):
                if i == index:
                    value = 0
                else:
                    value = (self.alph + score[t][i] - score_y) / self.tao
                    value = math.exp(value)
                loss += value
            loss = math.log(loss) * self.tao
            total_loss += loss
        total_loss = total_loss / len
        total_loss = torch.tensor(total_loss)
        total_loss = Variable(total_loss, requires_grad = True)
        return total_loss


if __name__ == '__main__':
    score = [0.876, 0.341, -0.221]
    score = torch.Tensor(score)
    y = torch.from_numpy(np.array([0],dtype=np.int64))
    y = F.one_hot(y, num_classes= 3).squeeze()
    loss_fn = my_loss_func(3,1,0.2)
    loss = loss_fn(score,y)


