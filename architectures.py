import torch
from torch.nn import Module
from attention_methods import Attention_channels
from attention_methods import Attention_timesteps
import math
import numpy as np     


class MLP_DeepSurv(Module):
    def __init__(self, d_input: int, d_agg_input: int, d_channel: int, d_output: int, 
                 d_other_input: int, device: str):
        super(MLP_DeepSurv, self).__init__() 
        """
        Multi-Layer Perceptron for DeepSurv-style survival analysis.
        Learns panel- and gene-level ctDNA features representations, and 
        combines pre-trained solid biopsy genomic-interaction representations, 
        and clinical features to obtain the log-risk of the Cox proportional hazards framework.
        
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

        # Batch normalization for the gene-level VAF features
        self.bn_input = torch.nn.BatchNorm1d((d_input+d_agg_input)*d_channel)
        # Batch normalization for the clinical features
        self.bn_other_input = torch.nn.BatchNorm1d(d_other_input)
        # Batch normalization for genomic-interaction representations pre-trained on solid biopsy DNA data
        self.bn_SB = torch.nn.BatchNorm1d((d_channel*2))         
        # MLP to learn a representation of the gene-level VAF features and panel-level features                                            
        self.mlp = torch.nn.Linear((d_input+d_agg_input)*d_channel, int((d_input+d_agg_input)*d_channel/2)) 
        # Fully connected layer to learn the log-risk of the Cox proportional hazards framework 
        # based on the learned ctDNA representation, clinical features, 
        # and pre-trained solid biopsy genomic-interaction representations
        self.fc_1 = torch.nn.Linear((d_channel*2)+int((d_input+d_agg_input)*d_channel/2)+d_other_input, d_output)
                                        
    def forward(self, x, x_agg, x_clinical, x_hidden_layer_SB): 
        
            """
            x: gene-level VAF features in the ctDNA data
            x_agg: panel-level ctDNA features
            x_clinical: clinical features
            x_hidden_layer_SB: pre-trained solid biopsy genomic-interaction representations
            """
            # Concatenate gene-level VAF features and panel-level features, 
            # and learn representations of the ctDNA data using batch normalization
            # and a Multi-Layer Perceptron (MLP) with a sigmoid activation function.
            x_xagg = torch.cat((x, x_agg), dim=1)   
            x_xagg = x_xagg.reshape(x_xagg.shape[0], -1)
            x_xagg = self.bn_input(x_xagg)
            x_xagg = self.mlp(x_xagg)           
            x_xagg = self.sigmoid(x_xagg) 
            # Pre-trained solid biopsy genomic-interaction representations are batch normalized                                       
            x_hidden_layer_SB = x_hidden_layer_SB.reshape(x_hidden_layer_SB.shape[0], -1)
            x_hidden_layer_SB = self.bn_SB(x_hidden_layer_SB)
            # Clinical features are batch normalized
            x_clinical = x_clinical.reshape(x_clinical.shape[0], -1)
            x_clinical = self.bn_other_input(x_clinical)
            # Concatenate the learned ctDNA representation, clinical features, 
            # and pre-trained solid biopsy genomic-interaction representations,
            # and learn the log-risk of the Cox proportional hazards framework 
            # through a fully connected layer with a sigmoid activation function.
            x_new = torch.cat((x_xagg, x_clinical, x_hidden_layer_SB), dim=-1)            
            x_new = self.dropout(x_new)
            x_new = self.fc_1(x_new)
            x_new = self.sigmoid(x_new) 
            return x_new          
    
    def predict(self, x, x_agg, x_clinical, x_hidden_layer_SB):
            # Idem the forward method, but without dropout for prediction   
            x_xagg = torch.cat((x, x_agg), dim=1)   
            x_xagg = x_xagg.reshape(x_xagg.shape[0], -1)
            x_xagg = self.bn_input(x_xagg)
            x_xagg = self.mlp(x_xagg)           
            x_xagg = self.sigmoid(x_xagg)                                                    
            x_hidden_layer_SB = x_hidden_layer_SB.reshape(x_hidden_layer_SB.shape[0], -1)
            x_hidden_layer_SB = self.bn_SB(x_hidden_layer_SB)            
            x_clinical = x_clinical.reshape(x_clinical.shape[0], -1)
            x_clinical = self.bn_other_input(x_clinical)            
            x_new = torch.cat((x_xagg, x_clinical, x_hidden_layer_SB), dim=-1)
            x_new = self.fc_1(x_new)
            x_new = self.sigmoid(x_new)         
            return x_new          


class GRU_timesteps(Module):
    def __init__(self, d_input: int, d_agg_input: int, d_channel: int, d_output: int, 
                 d_other_input: int, device: str):
        super(GRU_timesteps, self).__init__() 
        """
        Gated Recurrent Unit (GRU) for learning a representation of the gene-level VAF features 
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
        # GRU to learn a representation of the gene-level VAF features 
        # and panel-level features in the ctDNA data                                                          
        self.GRU = torch.nn.GRU(
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
            # and a Gated Recurrent Unit (GRU) keeping the last hidden layer as the 
            # learned representation of the ctDNA data.
            x_xagg = torch.cat((x, x_agg), dim=1)
            x_xagg = torch.transpose(x_xagg, -1, -2)
            x_xagg = self.bn_input(x_xagg)       
            x_GRU, _ = self.GRU(x_xagg)     
            x_xagg = x_GRU[:, -1, :]                      
            x_xagg = x_xagg.reshape(x_xagg.shape[0], -1)           
            x_xagg = self.sigmoid(x_xagg)             
            # Pre-trained solid biopsy genomic-interaction representations are batch normalized                                      
            x_hidden_layer_SB = x_hidden_layer_SB.reshape(x_hidden_layer_SB.shape[0], -1)
            x_hidden_layer_SB = self.bn_SB(x_hidden_layer_SB)
            # Clinical features are batch normalized    
            x_clinical = x_clinical.reshape(x_clinical.shape[0], -1)
            x_clinical = self.bn_other_input(x_clinical)
            # Concatenate the learned ctDNA representation, clinical features,
            # and pre-trained solid biopsy genomic-interaction representations,
            # and learn the log-risk of the Cox proportional hazards framework through 
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
            x_GRU, _ = self.GRU(x_xagg)     
            x_xagg = x_GRU[:, -1, :]                          
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
     

class Temporal_Transformer(Module):
    def __init__(self, d_input: int, d_agg_input: int, d_channel: int, 
                 d_output: int, d_other_input: int, device: str):
        super(Temporal_Transformer, self).__init__() 
        """
        Temporal Transformer for learning a representation of the gene-level VAF features 
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
        # Attention mechanism to learn a representation of the gene-level VAF features and 
        # panel-level features in the ctDNA data across longitudinal timepoints. 
        # In this model, temporal attention is learned, where each timestep is a token.   
        self.attention_timesteps = Attention_timesteps(d_input=d_input+d_agg_input,
                d_hidden=int((d_input+d_agg_input)/4), q=int((d_input+d_agg_input)/2), v=int((d_input+d_agg_input)/2), h=1, device=device)
        # Batch normalization for the clinical features                    
        self.bn_other_input = torch.nn.BatchNorm1d(d_other_input)
        # Batch normalization for genomic-interaction representations pre-trained on solid biopsy DNA data
        self.bn_SB = torch.nn.BatchNorm1d((d_channel*2))
        # Fully connected layer to learn the log-risk of the Cox proportional hazards framework 
        # based on the learned ctDNA representation, clinical features, and pre-trained solid 
        # biopsy genomic-interaction representations
        self.fc_1 = torch.nn.Linear((d_channel*2)+(int((d_input+d_agg_input)/4)*d_channel)+d_other_input, d_output)
                                
    def forward(self, x, x_agg, x_clinical, x_hidden_layer_SB):
            """
            x: gene-level VAF features in the ctDNA data
            x_agg: panel-level ctDNA features
            x_clinical: clinical features
            x_hidden_layer_SB: pre-trained solid biopsy genomic-interaction representations
            """
            # Concatenate gene-level VAF features and panel-level features, and learn 
            # representations of the ctDNA data across longitudinal timepoints using batch 
            # normalization and an attention mechanism where each timestep is a token.
            x_xagg = torch.cat((x, x_agg), dim=1)
            x_xagg = torch.transpose(x_xagg, -1, -2)
            x_xagg = self.bn_input(x_xagg)            
            x_xagg, score = self.attention_timesteps(x_xagg)                        
            x_xagg = x_xagg.reshape(x_xagg.shape[0], -1)           
            x_xagg = self.sigmoid(x_xagg) 
            # Batch normalization is applied to the pre-trained solid biopsy genomic-interaction 
            # representations and clinical features, and then they are concatenated 
            # with the learned ctDNA representations, and the log-risk of the Cox proportional hazards 
            # framework is learned through a fully connected layer with a sigmoid activation function.                            
            x_hidden_layer_SB = x_hidden_layer_SB.reshape(x_hidden_layer_SB.shape[0], -1)
            x_hidden_layer_SB = self.bn_SB(x_hidden_layer_SB)
            x_clinical = x_clinical.reshape(x_clinical.shape[0], -1)
            x_clinical = self.bn_other_input(x_clinical)
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
            x_xagg, score = self.attention_timesteps(x_xagg)                        
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
     

class Featurewise_Transformer(Module):
    def __init__(self, d_input: int, d_agg_input: int, d_channel: int, d_output: int, d_other_input: int,
                 device: str, dropout: float = 0.1):
        super(Featurewise_Transformer, self).__init__() 
        """
        Feature-wise Transformer for learning a representation of the gene-level VAF features 
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
        self.bn_input = torch.nn.BatchNorm1d(d_input+d_agg_input)
        # Attention mechanism to learn a representation of the gene-level VAF features and 
        # panel-level features in the ctDNA data across longitudinal timepoints. 
        # In this model, feature-wise attention is learned, where each ctDNA feature is a token.   
        self.attention_channels = Attention_channels(d_channel=d_channel,
                d_hidden=int(d_channel/2), q=d_channel, v=d_channel, h=1, device=device)
        # Batch normalization for the clinical features
        self.bn_other_input = torch.nn.BatchNorm1d(d_other_input)
        # Batch normalization for genomic-interaction representations pre-trained on solid biopsy DNA data
        self.bn_SB = torch.nn.BatchNorm1d((d_channel*2))
        # Fully connected layer to learn the log-risk of the Cox proportional hazards framework 
        # based on the learned ctDNA representations, clinical features, and 
        # pre-trained solid biopsy genomic-interaction representations
        self.fc_1 = torch.nn.Linear((d_channel*2)+((d_input+d_agg_input)*int(d_channel/2))+d_other_input, d_output)

                                
    def forward(self, x, x_agg, x_clinical, x_hidden_layer_SB):
            """
            x: gene-level VAF features in the ctDNA data
            x_agg: panel-level ctDNA features
            x_clinical: clinical features
            x_hidden_layer_SB: pre-trained solid biopsy genomic-interaction representations
            """
            # Concatenate gene-level VAF features and panel-level features, and learn
            # representations of the ctDNA data across longitudinal timepoints using batch
            # normalization and an attention mechanism where each ctDNA feature is a token.
            x_xagg = torch.cat((x, x_agg), dim=1)
            x_xagg = self.bn_input(x_xagg)            
            x_xagg, score = self.attention_channels(x_xagg)                        
            x_xagg = x_xagg.reshape(x_xagg.shape[0], -1)           
            x_xagg = self.sigmoid(x_xagg) 
            # Batch normalization is applied to the pre-trained solid biopsy genomic-interaction 
            # representations and clinical features, and then they are concatenated with the 
            # learned ctDNA representations, to output the log-risk of the Cox proportional hazards                                     
            x_hidden_layer_SB = x_hidden_layer_SB.reshape(x_hidden_layer_SB.shape[0], -1)
            x_hidden_layer_SB = self.bn_SB(x_hidden_layer_SB)
            x_clinical = x_clinical.reshape(x_clinical.shape[0], -1)
            x_clinical = self.bn_other_input(x_clinical)
            x_new = torch.cat((x_xagg, x_clinical, x_hidden_layer_SB), dim=-1)            
            x_new = self.dropout(x_new)
            x_new = self.fc_1(x_new)
            x_new = self.sigmoid(x_new) 
            return x_new          
    
    def predict(self, x, x_agg, x_clinical, x_hidden_layer_SB):
            # Idem the forward method, but without dropout for prediction
            x_xagg = torch.cat((x, x_agg), dim=1)
            x_xagg = self.bn_input(x_xagg)            
            x_xagg, score = self.attention_channels(x_xagg)                        
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
      

class SHERLOC(Module):
    def __init__(self, d_input: int, d_agg_input: int, d_channel: int, d_output: int, d_other_input: int,
                 device: str):
        super(SHERLOC, self).__init__() 
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
    
    def predict(self, x, x_agg, x_clinical, x_hidden_layer_SB):
            # Idem the forward method, but without dropout for prediction
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
            x_hidden_layer_SB = x_hidden_layer_SB.reshape(x_hidden_layer_SB.shape[0], -1)
            x_hidden_layer_SB = self.bn_SB(x_hidden_layer_SB) 
            x_clinical = x_clinical.reshape(x_clinical.shape[0], -1)
            x_clinical = self.bn_other_input(x_clinical)
            x_new = torch.cat((x, x_agg, x_clinical, x_hidden_layer_SB), dim=-1) 
            x_new = self.fc_1(x_new)
            x_new = self.sigmoid(x_new) 
            return x_new
                                            
    def explainability(self, x, x_agg, x_clinical, x_hidden_layer_SB):
            # Idem the predict method, but outputs the learned shared and feature-specific 
            # representations of the ctDNA data, clinical features, and pre-trained solid 
            # biopsy genomic-interaction representations, which can be used for explainability analyses.
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
            x_hidden_layer_SB = x_hidden_layer_SB.reshape(x_hidden_layer_SB.shape[0], -1)
            x_hidden_layer_SB = self.bn_SB(x_hidden_layer_SB)   
            x_clinical = x_clinical.reshape(x_clinical.shape[0], -1)
            x_clinical = self.bn_other_input(x_clinical)                                                                                      
            return x, x_agg, x_clinical, x_hidden_layer_SB


######################## Ablation models for the proposed method, SHERLOC ########################

class Proposed_Method_without_clinical(Module):
    def __init__(self, d_input: int, d_agg_input: int, d_channel: int, d_output: int, 
                 d_other_input: int, device: str, weights_solid_biopsy=None):
        super(Proposed_Method_without_clinical, self).__init__() 
        
        
        self.bn_input = torch.nn.BatchNorm1d(d_input)
        self.nb_agg = d_agg_input
        self.d_input = d_input
        self.d_channel = d_channel
                
        self.filter1 = torch.nn.Linear(d_channel,1)        
        self.maxpool = torch.nn.MaxPool1d(kernel_size=2, stride=2)                                
             
                         
        self.mlps = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(d_channel, 2)
            )
            for _ in range(d_agg_input)
        ])
                
        self.filter_SB = torch.nn.Linear(d_channel, int(d_channel/2))
        
        self.bn_x = torch.nn.BatchNorm1d(int(d_channel/2))
        self.bn_agg_input = torch.nn.BatchNorm1d(d_agg_input*2)
        self.bn_other_input = torch.nn.BatchNorm1d(d_other_input)
        self.bn_SB = torch.nn.BatchNorm1d((d_channel*2))
        self.fc_1 = torch.nn.Linear((d_channel*2)+(d_input)+int(2*d_agg_input), d_output)


        self.sigmoid = torch.nn.Sigmoid()
        self.dropout = torch.nn.Dropout(p=0.4)
        
        self.weights_solid_biopsy = weights_solid_biopsy
                        
    def forward(self, x, x_agg, x_clinical, x_hidden_layer_SB):
            
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
                                   
            x_hidden_layer_SB = x_hidden_layer_SB.reshape(x_hidden_layer_SB.shape[0], -1)
            x_hidden_layer_SB = self.bn_SB(x_hidden_layer_SB)
                        
            x_new = torch.cat((x, x_agg, x_hidden_layer_SB), dim=-1)
            x_new = self.dropout(x_new)
            x_new = self.fc_1(x_new)
            x_new = self.sigmoid(x_new) 
        
            return x_new          
    
    def predict(self, x, x_agg, x_clinical, x_hidden_layer_SB):
               
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
                                   
            x_hidden_layer_SB = x_hidden_layer_SB.reshape(x_hidden_layer_SB.shape[0], -1)
            x_hidden_layer_SB = self.bn_SB(x_hidden_layer_SB)
                        
            x_new = torch.cat((x, x_agg, x_hidden_layer_SB), dim=-1)            
            x_new = self.fc_1(x_new)
            x_new = self.sigmoid(x_new) 
        
            return x_new
                                    
    def extract_last_hidden_layer(self, x, x_agg, x_clinical, x_hidden_layer_SB):
        
        with torch.no_grad():
                    
            x = self.bn_input(x)                            
            x = self.filter1(x)
            x = self.sigmoid(x)                        
            x = x.reshape(x.shape[0], -1)                                              
            return x    
        


class Proposed_Method_without_Pretraining(Module):
    def __init__(self, d_input: int, d_agg_input: int, d_channel: int, d_output: int, d_hidden: int, d_other_input: int,
                 nb_filters: int, device: str, nb_genes: int, dropout: float = 0.1, weights_solid_biopsy=None):
        super(Proposed_Method_without_Pretraining, self).__init__() 
        
        
        self.bn_input = torch.nn.BatchNorm1d(d_input)
        self.nb_agg = d_agg_input
        self.d_input = d_input
        self.d_channel = d_channel                
        self.filter1 = torch.nn.Linear(d_channel,1)
                                                 
        self.mlps = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(d_channel, 2)
            )
            for _ in range(d_agg_input)
        ])
                
        self.filter_SB = torch.nn.Linear(d_channel, int(d_channel/2))
        
        self.bn_x = torch.nn.BatchNorm1d(int(d_channel/2))
        self.bn_agg_input = torch.nn.BatchNorm1d(d_agg_input*2)
        self.bn_other_input = torch.nn.BatchNorm1d(d_other_input)
        self.bn_SB = torch.nn.BatchNorm1d((d_channel*2))
        self.bn_pertimestep = torch.nn.BatchNorm1d(d_channel) 

        self.per_timestep_encoder = torch.nn.Linear(nb_genes,2) 
        if weights_solid_biopsy is not None:
            self.per_timestep_encoder.load_state_dict(weights_solid_biopsy)
               
        self.fc_1 = torch.nn.Linear((d_channel*2)+(d_input)+int(2*d_agg_input)+d_other_input, d_output)
        self.sigmoid = torch.nn.Sigmoid()
        self.dropout = torch.nn.Dropout(p=0.4)        
        self.weights_solid_biopsy = weights_solid_biopsy
                        
    def forward(self, x, x_agg, x_clinical, x_per_timestep_encoding):
            
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
                                   
            x_per_timestep_encoding = x_per_timestep_encoding.transpose(-1, -2)
            x_per_timestep_encoding = self.bn_pertimestep(x_per_timestep_encoding)
            x_per_timestep_encoding = self.per_timestep_encoder(x_per_timestep_encoding)
            x_per_timestep_encoding = self.sigmoid(x_per_timestep_encoding) 
            x_per_timestep_encoding = x_per_timestep_encoding.reshape(x_per_timestep_encoding.shape[0], -1)
                         
            x_clinical = x_clinical.reshape(x_clinical.shape[0], -1)
            x_clinical = self.bn_other_input(x_clinical)
            
            x_new = torch.cat((x, x_agg, x_clinical, x_per_timestep_encoding), dim=-1)
            x_new = self.dropout(x_new)
            x_new = self.fc_1(x_new)
            x_new = self.sigmoid(x_new) 
        
            return x_new          
    
    def predict(self, x, x_agg, x_clinical, x_per_timestep_encoding):
               
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
                                   
            x_per_timestep_encoding = x_per_timestep_encoding.transpose(-1, -2)
            x_per_timestep_encoding = self.bn_pertimestep(x_per_timestep_encoding)
            x_per_timestep_encoding = self.per_timestep_encoder(x_per_timestep_encoding)
            x_per_timestep_encoding = self.sigmoid(x_per_timestep_encoding) 
            x_per_timestep_encoding = x_per_timestep_encoding.reshape(x_per_timestep_encoding.shape[0], -1)
                        
            x_clinical = x_clinical.reshape(x_clinical.shape[0], -1)
            x_clinical = self.bn_other_input(x_clinical)
            
            x_new = torch.cat((x, x_agg, x_clinical, x_per_timestep_encoding), dim=-1)
            x_new = self.fc_1(x_new)
            x_new = self.sigmoid(x_new) 
        
            return x_new 
                 
                                    
    def extract_last_hidden_layer(self, x, x_agg, x_clinical, x_per_timestep_encoding):
        
        with torch.no_grad():
                    
            x = self.bn_input(x)                            
            x = self.filter1(x)
            x = self.sigmoid(x)                        
            x = x.reshape(x.shape[0], -1)                                              
            return x    
        



class Proposed_Method_without_ctDNA_dynamics_encoder(Module):
    def __init__(self, d_input: int, d_agg_input: int, d_channel: int, d_output: int, d_hidden: int, d_other_input: int,
                 nb_filters: int, device: str, dropout: float = 0.1, weights_solid_biopsy=None):
        super(Proposed_Method_without_ctDNA_dynamics_encoder, self).__init__() 
        
        
        self.bn_input = torch.nn.BatchNorm1d(d_input)
        self.nb_agg = d_agg_input
        self.d_input = d_input
        self.d_channel = d_channel
                
        self.filter1 = torch.nn.Linear(d_channel,1)
                         
        self.mlps = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(d_channel, 2)
            )
            for _ in range(d_agg_input)
        ])
        
        self.filter_SB = torch.nn.Linear(d_channel, int(d_channel/2))        
        self.bn_x = torch.nn.BatchNorm1d(int(d_channel/2))
        self.bn_agg_input = torch.nn.BatchNorm1d(d_agg_input*2)
        self.bn_other_input = torch.nn.BatchNorm1d(d_other_input)
        self.bn_SB = torch.nn.BatchNorm1d((d_channel*2))

        self.fc_1 = torch.nn.Linear((d_channel*2)+d_other_input, d_output)
        self.sigmoid = torch.nn.Sigmoid()
        self.dropout = torch.nn.Dropout(p=0.4)
        
        self.weights_solid_biopsy = weights_solid_biopsy
                        
    def forward(self, x, x_agg, x_clinical, x_hidden_layer_SB):
            
            x_hidden_layer_SB = x_hidden_layer_SB.reshape(x_hidden_layer_SB.shape[0], -1)
            x_hidden_layer_SB = self.bn_SB(x_hidden_layer_SB)
            
            x_clinical = x_clinical.reshape(x_clinical.shape[0], -1)
            x_clinical = self.bn_other_input(x_clinical)
            
            x_new = torch.cat((x_clinical, x_hidden_layer_SB), dim=-1)
            x_new = self.dropout(x_new)
            x_new = self.fc_1(x_new)
            x_new = self.sigmoid(x_new) 
        
            return x_new
          
    
    def predict(self, x, x_agg, x_clinical, x_hidden_layer_SB):
               
            x_hidden_layer_SB = x_hidden_layer_SB.reshape(x_hidden_layer_SB.shape[0], -1)
            x_hidden_layer_SB = self.bn_SB(x_hidden_layer_SB)
            
            x_clinical = x_clinical.reshape(x_clinical.shape[0], -1)
            x_clinical = self.bn_other_input(x_clinical)
            
            x_new = torch.cat((x_clinical, x_hidden_layer_SB), dim=-1)
            x_new = self.fc_1(x_new)
            x_new = self.sigmoid(x_new) 
        
            return x_new
                                            

class Proposed_Method_without_Per_timestep_ctDNA_encoder(Module):
    def __init__(self, d_input: int, d_agg_input: int, d_channel: int, d_output: int, d_hidden: int, d_other_input: int,
                 nb_filters: int, device: str, dropout: float = 0.1, weights_solid_biopsy=None):
        super(Proposed_Method_without_Per_timestep_ctDNA_encoder, self).__init__() 
        
        
        self.bn_input = torch.nn.BatchNorm1d(d_input)
        self.nb_agg = d_agg_input
        self.d_input = d_input
        self.d_channel = d_channel
                
        self.filter1 = torch.nn.Linear(d_channel,1)
                                         
        self.mlps = torch.nn.ModuleList([
            torch.nn.Sequential(
                torch.nn.Linear(d_channel, 2)
            )
            for _ in range(d_agg_input)
        ])
                
        self.filter_SB = torch.nn.Linear(d_channel, int(d_channel/2))        
        self.bn_x = torch.nn.BatchNorm1d(int(d_channel/2))
        self.bn_agg_input = torch.nn.BatchNorm1d(d_agg_input*2)
        self.bn_other_input = torch.nn.BatchNorm1d(d_other_input)
        self.bn_SB = torch.nn.BatchNorm1d((d_channel*2))
        self.fc_1 = torch.nn.Linear((d_input)+int(2*d_agg_input)+d_other_input, d_output)


        self.sigmoid = torch.nn.Sigmoid()
        self.dropout = torch.nn.Dropout(p=0.4)        
        self.weights_solid_biopsy = weights_solid_biopsy
                        
    def forward(self, x, x_agg, x_clinical, x_hidden_layer_SB):
            
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
                                               
            x_clinical = x_clinical.reshape(x_clinical.shape[0], -1)
            x_clinical = self.bn_other_input(x_clinical)
            
            x_new = torch.cat((x, x_agg, x_clinical), dim=-1)
            x_new = self.dropout(x_new)
            x_new = self.fc_1(x_new)
            x_new = self.sigmoid(x_new) 
        
            return x_new
         
    
    def predict(self, x, x_agg, x_clinical, x_hidden_layer_SB):
               
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
            
            x_clinical = x_clinical.reshape(x_clinical.shape[0], -1)
            x_clinical = self.bn_other_input(x_clinical)
            
            x_new = torch.cat((x, x_agg, x_clinical), dim=-1)
            x_new = self.fc_1(x_new)
            x_new = self.sigmoid(x_new) 
        
            return x_new
                                    
    def extract_last_hidden_layer(self, x, x_agg, x_clinical, x_hidden_layer_SB):
        
        with torch.no_grad():
                    
            x = self.bn_input(x)                            
            x = self.filter1(x)
            x = self.sigmoid(x)                        
            x = x.reshape(x.shape[0], -1)                                              
            return x    
        
        

class MLP_SB(Module):
    def __init__(self, d_input: int, d_channel: int, d_output: int, d_other_input: int,
                         device: str, dropout: float = 0.1):
        super(MLP_SB, self).__init__() 
            
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
    
 
