import torch
from torch.nn import Module    

class SHERLOC(Module):
    """
        Proposed model, SHERLOC (Shared and Exclusive Representations of Longitudinal ctDNA data), 
        which adopts a hybrid strategy learning shared longitudinal representations 
        of the gene-level VAF features and feature-specific temporal representations for 
        panel-level ctDNA features. These representations are then combined with pre-trained 
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
    def __init__(self, d_input: int, d_agg_input: int, d_channel: int, d_output: int, d_other_input: int,
                 device: str):
        super(SHERLOC, self).__init__() 
        
        self.d_input = d_input
        self.nb_agg = d_agg_input
        self.d_channel = d_channel
        self.sigmoid = torch.nn.Sigmoid()
        self.dropout = torch.nn.Dropout(p=0.4)
        # Batch normalization for the gene-level VAF features and panel-level features    
        self.bn_input = torch.nn.BatchNorm1d(d_input)
        # Shared representation of the gene-level VAF features in the ctDNA data 
        # across longitudinal timepoints.                  
        self.filter1 = torch.nn.Linear(d_channel,1)                        
        # Feature-specific temporal representations for panel-level ctDNA features 
        # across longitudinal timepoints, where each panel-level feature has a specific MLP 
        # to learn its temporal representation.
        self.mlps = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(d_channel, 2)
            )
            for _ in range(d_agg_input)
        ])
        
        # Batch normalization for the clinical features and pre-trained solid biopsy genomic-interaction 
        # representations, and for the learned representations of the panel-level ctDNA features across 
        # longitudinal timepoints.
        self.bn_agg_input = torch.nn.BatchNorm1d(d_agg_input*2)
        self.bn_other_input = torch.nn.BatchNorm1d(d_other_input)
        self.bn_SB = torch.nn.BatchNorm1d((d_channel*2))
        # Fully connected layer to learn the log-risk of the Cox proportional hazards framework based on 
        # the learned shared and feature-specific representations of the ctDNA data, clinical features, 
        # and pre-trained solid biopsy genomic-interaction representations 
        self.fc_1 = torch.nn.Linear((d_channel*2)+(d_input)+int(2*d_agg_input)+d_other_input, d_output)
          
    def forward(self, x, x_agg, x_clinical, x_hidden_layer_SB):
            """
            x: gene-level VAF features in the ctDNA data
            x_agg: panel-level ctDNA features
            x_clinical: clinical features
            x_hidden_layer_SB: pre-trained solid biopsy genomic-interaction representations
            """
            # Learn shared longitudinal representations of the gene-level VAF features, and 
            # feature-specific temporal representations for panel-level ctDNA features 
            # across longitudinal timepoints.
            x = self.bn_input(x)                            
            x = self.filter1(x)
            x = self.sigmoid(x)
            x = x.reshape(x.shape[0], -1)      
            x_temp = []
            for feature in range(self.nb_agg):
                x_feature = x_agg[:, feature, :]  
                mlp = self.mlps[feature]
                output = mlp(x_feature)  
                x_temp.append(output.unsqueeze(1))     
            x_agg = torch.cat(x_temp, dim=1)  
            x_agg = x_agg.reshape(x_agg.shape[0], -1)
            x_agg = self.bn_agg_input(x_agg)
            # Batch normalization is applied to the pre-trained solid biopsy genomic-interaction 
            # representations and clinical features, and then they are concatenated with the learned 
            # shared and feature-specific representations of the ctDNA data, to output the log-risk of the 
            # Cox proportional hazards framework through a fully connected layer with a sigmoid activation.                       
            x_hidden_layer_SB = x_hidden_layer_SB.reshape(x_hidden_layer_SB.shape[0], -1)
            x_hidden_layer_SB = self.bn_SB(x_hidden_layer_SB)
            x_clinical = x_clinical.reshape(x_clinical.shape[0], -1)
            x_clinical = self.bn_other_input(x_clinical)
            x_new = torch.cat((x, x_agg, x_clinical, x_hidden_layer_SB), dim=-1)
            x_new = self.dropout(x_new)
            x_new = self.fc_1(x_new)
            x_new = self.sigmoid(x_new) 
            return x_new          
    
