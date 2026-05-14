import torch
from torch.nn import Module   

class MLP_SolidBiopsy(Module):
    """
        Multi-Layer Perceptron for learning representations of the genomic interactions 
        in the solid biopsy DNA data (e.g., MSK-CHORD), and combining it with clinical 
        features to obtain the log-risk of the Cox proportional hazards framework. 
        
        Extracted representations from the last hidden layer of this MLP are used as the 
        pre-trained solid biopsy genomic-interaction representations in all benchmarked 
        models.
        

        Args:
            d_input (int): Number of input features (e.g., number of genes) for the genomic 
            interactions in the solid biopsy DNA data.
            d_representations (int): Number of representations to learn for the genomic 
            interactions in the solid biopsy DNA data.
            d_output (int): Number of output features (1 for log-risk).
            d_other_input (int): Number of other input features (clinical features).
            device (str): Device to run the model on ('cpu' or 'cuda').
            dropout (float): Dropout rate for regularization.
    """
    def __init__(self, d_input: int, d_representations: int, d_output: int, d_other_input: int,
                         device: str, dropout: float = 0.1):
        super(MLP_SolidBiopsy, self).__init__() 
            
        self.bn_input = torch.nn.BatchNorm1d(d_input)        
        self.fc1 = torch.nn.Linear(d_input,d_representations)
        self.bn1 = torch.nn.BatchNorm1d(d_representations+d_other_input)
        self.fc2 = torch.nn.Linear(d_representations+d_other_input, d_output)                        
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
    
 

