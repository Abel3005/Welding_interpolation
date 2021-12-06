import torch
import torch.nn.functional as F
from torch.utils.data import dataloader
from model import LinearRegressionModel
from dataset import InterpDataset
from config import Config
import numpy as np
from torch.utils.data import DataLoader

def main():
    model = LinearRegressionModel()
    config = Config()
    dataset = InterpDataset(testflag=True)
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    
    model.load_state_dict(torch.load(config.bestweigtfile))
    for p in model.linear.parameters():
        print(p)
    model.eval()
    for batch_idx, samples in enumerate(dataloader):
        x_train, y_train = samples
        prediction = model(x_train)
        cost = F.mse_loss(prediction, y_train)
        print('Batch {}/{} Cost: {:.6f}'.format(
             batch_idx+1, len(dataloader), cost.item()
            ))
    regression_matrix=np.array([
            [-2.3043e-04, -1.5157e-01, -7.8664e-03],
            [4.4900e-03, -1.3388e-01, -1.4054e-01],
            [-1.7953e-02,  3.8481e-01, -2.0433e-02]  
        ])
    result = regression_matrix@np.array([0.733481,0.104318,0.592396]) + np.array([0.0219,  0.0956, -0.0115])
    print(result)

if __name__ == '__main__':
    main()