from torch.utils.data import Dataset
from config import Config
import csv
import torch

config = Config()

class InterpDataset(Dataset):
    
    def make_lookup_table(self, num_points):
        self.x_data = torch.zeros((num_points,3))
        self.y_data = torch.zeros((num_points,3))
        for i in range(0,num_points):
            key_point = torch.FloatTensor([float(self.lookup_list[i][1]), float(self.lookup_list[i][2]), float(self.lookup_list[i][3])])
            err_trans = torch.FloatTensor([float(self.lookup_list[i][4]), float(self.lookup_list[i][5]), float(self.lookup_list[i][6])])
            self.x_data[i] = key_point
            self.y_data[i] = err_trans
            
    
    def make_lookup_list(self):
        filename = None
        if self.testflag:
            filename = config.testfile
        else:
            filename = config.lookupfile
        csvfile = open(filename, 'r')
        self.lookup_list = []
        self.test_list = []
        rdr = csv.reader(csvfile)
        for line in rdr:
            self.lookup_list.append(line)
        return
    
    def __init__(self, testflag=False):
        self.testflag = testflag
        self.make_lookup_list()
        self.make_lookup_table(len(self.lookup_list))
        
    def __len__(self):
        return len(self.x_data)
    
    def __getitem__(self, idx):
        x = torch.FloatTensor(self.x_data[idx])      
        y = torch.FloatTensor(self.y_data[idx])      
        return x ,y 