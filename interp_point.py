from config import Config
from model import LinearRegressionModel
import torch.nn.functional as F
from dataset import InterpDataset
from torch.utils.data import DataLoader
import torch



def main():
    config = Config()
    model = LinearRegressionModel()
    dataset = InterpDataset()
    dataloader = DataLoader(dataset, batch_size=2, shuffle=True)
    num_epoch = config.epochs
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    best_cost  = 10.0
    for epoch in range(num_epoch + 1):
       for batch_idx, samples in enumerate(dataloader):
           x_train, y_train = samples
           prediction = model(x_train)
           cost = F.mse_loss(prediction, y_train)
           
           optimizer.zero_grad()
           cost.backward()
           optimizer.step()
           if  epoch % 100 == 0:
            print('Epoch {:4d}/{} Batch {}/{} Cost: {:.6f}'.format(
                epoch, num_epoch, batch_idx+1, len(dataloader), cost.item()
            ))
            if best_cost > cost.item():
                best_cost = cost.item()
                torch.save(model.state_dict(),config.bestweigtfile)
                
if __name__ == '__main__':
    main()