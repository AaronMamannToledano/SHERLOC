import torch
from torch.nn import Module   

class MLP_SolidBiopsy(Module):
    def __init__(self, d_input: int, d_channel: int, d_output: int, d_other_input: int,
                         device: str, dropout: float = 0.1):
        super(MLP_SolidBiopsy, self).__init__() 
            
        self.bn_input = torch.nn.BatchNorm1d(d_input)        
        self.fc1 = torch.nn.Linear(d_input,2)
        self.bn1 = torch.nn.BatchNorm1d(2+d_other_input)
        self.fc2 = torch.nn.Linear(2+d_other_input, d_output)                        
        self.sigmoid = torch.nn.Sigmoid()
        self.dropout = torch.nn.Dropout(p=dropout)
        
    def forward(self, x, x_clinical):
                
        x = x.reshape(x.shape[0], -1)
        x = self.bn_input(x)
        x = self.fc1(x)
        x = self.sigmoid(x) 
        x_clinical = x_clinical.reshape(x_clinical.shape[0], -1)
        x_new = torch.cat((x, x_clinical), dim=-1)
        x_new = self.bn1(x_new)
        x_new = self.dropout(x_new)            
        x_new = self.fc2(x_new)
        x_new = self.sigmoid(x_new) 
        return x_new  

    def extract_encoder(self, x):            
        with torch.no_grad():    
            x = x.reshape(x.shape[0], -1)
            x = self.bn_input(x) 
            x = self.fc1(x)
            x = self.sigmoid(x) 
            return x  
    
 

