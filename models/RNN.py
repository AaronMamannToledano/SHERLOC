import torch
from torch.nn import Module   

class RNN_timesteps(Module):
    def __init__(self, d_input: int, d_agg_input: int, d_channel: int, d_output: int, d_hidden: int, d_other_input: int,
                 nb_filters: int, device: str, dropout: float = 0.1, weights_solid_biopsy=None):
        super(RNN_timesteps, self).__init__() 
        """
        Recurrent Neural Network (RNN) for learning a representation of the gene-level VAF features 
        and panel-level features in the ctDNA data, and combining it with pre-trained 
        solid biopsy genomic-interaction representations, and clinical features to obtain 
        the log-risk of the Cox proportional hazards framework.
                
        Args:
            d_input (int): Number of input genes for the gene-level VAF features in the ctDNA data.
            d_agg_input (int): Number of panel-level ctDNA features.
            d_channel (int): Number of longitudinal ctDNA measurments per patient.
            d_output (int): Number of output features (1 for log-risk).
            d_other_input (int): Number of other input features (clinical features).
            device (str): Device to run the model on ('cpu' or 'cuda').

        """         
        self.d_input = d_input
        self.d_channel = d_channel
        self.sigmoid = torch.nn.Sigmoid()
        self.dropout = torch.nn.Dropout(p=0.3)
        # Batch normalization for the gene-level VAF features and panel-level features
        self.bn_input = torch.nn.BatchNorm1d(d_channel)
        # Recurrent Neural Network (RNN) to learn a representation of the gene-level VAF features and 
        # panel-level features in the ctDNA data                                               
        self.rnn = torch.nn.RNN(
        input_size=d_input+d_agg_input,   
        hidden_size=int((d_input+d_agg_input)),  
        num_layers=1,
        batch_first=True
        )
        # Batch normalization for the clinical features
        self.bn_other_input = torch.nn.BatchNorm1d(d_other_input)
        # Batch normalization for genomic-interaction representations pre-trained on solid biopsy DNA data
        self.bn_SB = torch.nn.BatchNorm1d((d_channel*2))
        # Fully connected layer to learn the log-risk of the Cox proportional hazards framework 
        # based on the learned ctDNA representation, clinical features, and pre-trained solid biopsy 
        # genomic-interaction representations
        self.fc_1 = torch.nn.Linear((d_channel*2)+int((d_input+d_agg_input))+d_other_input, d_output)
                      
    def forward(self, x, x_agg, x_clinical, x_hidden_layer_SB):
            """
            x: gene-level VAF features in the ctDNA data
            x_agg: panel-level ctDNA features
            x_clinical: clinical features
            x_hidden_layer_SB: pre-trained solid biopsy genomic-interaction representations
            """
            # Concatenate gene-level VAF features and panel-level features, 
            # and learn representations of the ctDNA data using batch normalization
            # and a Recurrent Neural Network (RNN) keeping the last hidden layer as the 
            # learned representation of the ctDNA data.
            x_xagg = torch.cat((x, x_agg), dim=1)
            x_xagg = torch.transpose(x_xagg, -1, -2)
            x_xagg = self.bn_input(x_xagg)   
            x_rnn, _ = self.rnn(x_xagg)     
            x_xagg = x_rnn[:, -1, :]                 
            x_xagg = x_xagg.reshape(x_xagg.shape[0], -1)           
            x_xagg = self.sigmoid(x_xagg) 
            # Pre-trained solid biopsy genomic-interaction representations are batch normalized                                
            x_hidden_layer_SB = x_hidden_layer_SB.reshape(x_hidden_layer_SB.shape[0], -1)
            x_hidden_layer_SB = self.bn_SB(x_hidden_layer_SB)
            # Clinical features are batch normalized
            x_clinical = x_clinical.reshape(x_clinical.shape[0], -1)
            x_clinical = self.bn_other_input(x_clinical)
            # Concatenate the learned ctDNA representation, clinical features, and 
            # pre-trained solid biopsy genomic-interaction representations, and 
            # learn the log-risk of the Cox proportional hazards framework through 
            # a fully connected layer with a sigmoid activation function.
            x_new = torch.cat((x_xagg, x_clinical, x_hidden_layer_SB), dim=-1)            
            x_new = self.dropout(x_new)
            x_new = self.fc_1(x_new)
            x_new = self.sigmoid(x_new) 
            return x_new          
    
    def predict(self, x, x_agg, x_clinical, x_hidden_layer_SB):
            # Idem the forward method, but without dropout for prediction
            x_xagg = torch.cat((x, x_agg), dim=1)
            x_xagg = torch.transpose(x_xagg, -1, -2)
            x_xagg = self.bn_input(x_xagg)   
            x_rnn, _ = self.rnn(x_xagg)     
            x_xagg = x_rnn[:, -1, :]                  
            x_xagg = x_xagg.reshape(x_xagg.shape[0], -1)           
            x_xagg = self.sigmoid(x_xagg)                                           
            x_hidden_layer_SB = x_hidden_layer_SB.reshape(x_hidden_layer_SB.shape[0], -1)
            x_hidden_layer_SB = self.bn_SB(x_hidden_layer_SB)   
            x_clinical = x_clinical.reshape(x_clinical.shape[0], -1)
            x_clinical = self.bn_other_input(x_clinical)    
            x_new = torch.cat((x_xagg, x_clinical, x_hidden_layer_SB), dim=-1)
            x_new = self.fc_1(x_new)
            x_new = self.sigmoid(x_new) 
            return x_new          
     
