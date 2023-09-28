import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from transformer import Config, TransformerEncoder, CrossEncoder
from decoder import DecoderConfig, DecoderModel

# import spikingjelly
from spikingjelly.activation_based import neuron, functional, surrogate, layer


class MULTModel(nn.Module):
    def __init__(self):
        """
        Construct a MulT model.
        """
        super(MULTModel, self).__init__()
        self.orig_d_l, self.orig_d_a, self.orig_d_v = 300, 74, 35
        self.d_l, self.d_a, self.d_v = 30, 30, 30
        self.vonly = True
        self.aonly = True
        self.lonly = True
        self.num_heads = 5
        self.layers = 1
        print(f'layers: {self.layers}')
        self.attn_dropout = 0.1
        self.attn_dropout_a = 0.0
        self.attn_dropout_v = 0.0
        #self.relu_dropout = 0.1
        #self.res_dropout = 0.1
        self.out_dropout = 0.0
        self.embed_dropout = 0.25
        #self.attn_mask = True
        output_dim = 1        # This is actually not a hyperparameter :-)
        
        self.nclasses = 6

        self.cross_entropy = 1

        # 1. Temporal convolutional layers
        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=1, padding=0, bias=False)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=1, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=1, padding=0, bias=False)

        # 2. Self Attentions
        self.trans_l_with_self = self.get_network(self_type='l')
        self.trans_a_with_self = self.get_network(self_type='a')
        self.trans_v_with_self = self.get_network(self_type='v')

        """"""""""""""""""
        self.T = 5
        self.snna = snn()
        self.snnv = snn()
        self.trans_snna = self.get_network(self_type='snna')
        self.trans_snnv = self.get_network(self_type='snnv')
        """"""""""""""""""

        # 3. Crossmodal Attentions
        self.trans_a_with_v = self.get_network(self_type='av')
        self.trans_v_with_a = self.get_network(self_type='va')

        self.trans_l_with_av = self.get_network(self_type='lav')
        self.trans_l_with_va = self.get_network(self_type='lva')
        self.trans_av_with_l = self.get_network(self_type='avl')
        self.trans_va_with_l = self.get_network(self_type='val')

        # decoder
        self.decoder_config = self.get_decoder_config()
        self.decoder        = DecoderModel(self.decoder_config)

        self.classifier = EmotionClassifier(self.d_l, 1)

    def get_network(self, self_type='l', layers=-1):
        if self_type in ['l', 'avl', 'val', 'lav', 'lva']:
            embed_dim, attn_dropout = self.d_l, self.attn_dropout
        elif self_type in ['a', 'va', 'snna']:
            embed_dim, attn_dropout = self.d_a, self.attn_dropout_a
        elif self_type in ['v', 'av', 'snnv']:
            embed_dim, attn_dropout = self.d_v, self.attn_dropout_v
        else:
            raise ValueError("Unknown network type")

        config = Config(embed_dim, num_hidden_layers=max(layers,self.layers), num_attention_heads=self.num_heads, intermediate_size=4*embed_dim, 
                        attention_probs_dropout_prob=attn_dropout,
                        )
        
        return TransformerEncoder(config)
    
    def get_decoder_config(self):
        return DecoderConfig( 
                            attention_probs_dropout_prob = 0.3,
                            hidden_act                   = "gelu",
                            hidden_dropout_prob            = 0.3,
                            hidden_size                    = 30,
                            initializer_range              = 0.02,
                            intermediate_size              = 256,
                            num_attention_heads            = 5,
                            #num_hidden_layers              = 3,
                            type_vocab_size                = 2,
                            vocab_size_or_config_json_file = 6,
                            num_decoder_layers             = 1,
                            max_target_embeddings          = 6,)
            
    def forward(self, x_l, x_a, x_v):
        """
        text, audio, and vision should have dimension [batch_size, seq_len, n_features]
        """
        bsz = x_l.shape[0]
                                                                                           # torch.Size([128, 60, 300])
        x_l = F.dropout(x_l.transpose(1, 2), p=self.embed_dropout, training=self.training) # torch.Size([128, 300, 60])
        x_a = x_a.transpose(1, 2)
        x_v = x_v.transpose(1, 2)
        
        # Project the textual/visual/audio features
        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)  # torch.Size([128, 30, 60])
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)

        proj_x_a = proj_x_a.permute(0, 2, 1)
        proj_x_v = proj_x_v.permute(0, 2, 1)
        proj_x_l = proj_x_l.permute(0, 2, 1)  # torch.Size([128, 60, 30])      

        # self
        h_l = self.trans_l_with_self(proj_x_l, proj_x_l, proj_x_l)[0][-1]
        h_a = self.trans_a_with_self(proj_x_a, proj_x_a, proj_x_a)[0][-1]
        h_v = self.trans_v_with_self(proj_x_v, proj_x_v, proj_x_v)[0][-1]

        """"""""""""""""""
        proj_x_a_flat = proj_x_a.reshape((proj_x_a.size(0)*proj_x_a.size(1), proj_x_a.size(2)))
        proj_x_v_flat = proj_x_v.reshape((proj_x_v.size(0)*proj_x_v.size(1), proj_x_v.size(2)))
        temp_cross_entropy = F.cross_entropy(proj_x_a_flat, torch.argmax(proj_x_v_flat, dim=1))
        temp_cross_entropy = temp_cross_entropy.item()
        # snn_threshold = abs(temp_cross_entropy - self.cross_entropy) / max(temp_cross_entropy, self.cross_entropy)
        snn_threshold = temp_cross_entropy / self.cross_entropy
        self.cross_entropy = temp_cross_entropy
        """"""""""""""""""
        """"""""""""""""""
        # torch.Size([128, 60, 30])
        snn_a1 = self.snna(proj_x_a, snn_threshold)
        for t in range(1, self.T):
            snn_a1 += self.snna(proj_x_a, snn_threshold)
        snn_a1 = snn_a1/self.T

        snn_v1 = self.snnv(proj_x_v, snn_threshold)
        for t in range(1, self.T):
            snn_v1 += self.snnv(proj_x_v, snn_threshold)
        snn_v1 = snn_v1/self.T

        snn_a1_smx = nn.functional.softmax(snn_a1,dim=0)
        # h_a_snn = h_a*snn_a1_smx  
        h_a_snn = h_a + h_a*snn_a1_smx  
        h_a_snn_trm = h_a + self.trans_snna(h_a, h_a_snn, h_a_snn)[0][-1]

        snn_v1_smx = nn.functional.softmax(snn_v1,dim=0)
        # h_v_snn = h_v * snn_v1_smx  
        h_v_snn = h_v + h_v * snn_v1_smx  
        h_v_snn_trm = h_v + self.trans_snnv(h_v, h_v_snn, h_v_snn)[0][-1]
        """"""""""""""""""

        # a & v
        h_av = self.trans_a_with_v(h_a_snn_trm, h_v_snn_trm, h_v_snn_trm)[0][-1]
        h_va = self.trans_v_with_a(h_v_snn_trm, h_a_snn_trm, h_a_snn_trm)[0][-1]

        # with l
        h_lav = self.trans_l_with_av(proj_x_l, h_av, h_av)[0][-1]
        h_lva = self.trans_l_with_va(proj_x_l, h_va, h_va)[0][-1]
        h_avl = self.trans_av_with_l(h_av, proj_x_l, proj_x_l)[0][-1]
        h_val = self.trans_va_with_l(h_va, proj_x_l, proj_x_l)[0][-1]

        
        sequence = torch.cat([h_lav, h_lva, h_avl, h_val, h_l, h_a, h_v], dim=1)
        label_input    = torch.tensor(np.arange(self.nclasses)).to(device=x_l.device).unsqueeze(0)
        
        label_masks    = torch.ones(bsz, len(label_input)).to(device=x_l.device)
        sequence_masks = torch.ones(sequence.shape[:-1]).to(x_l.device)
        last_hs_proj   = self.decoder(label_input.to(dtype=torch.long), sequence, label_masks, sequence_masks)
        
        output = self.classifier(last_hs_proj).view(-1,self.nclasses)
        return output


class EmotionClassifier(nn.Module):
    def __init__(self, input_dims, num_classes=1, dropout=0.1):
        super(EmotionClassifier, self).__init__()
        self.dense = nn.Linear(input_dims, num_classes)
        self.activation = nn.Sigmoid()
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, seq_input):

        output = self.dense(seq_input)
        output = self.dropout(output)
        output = self.activation(output)

        return output

class snn(nn.Module):
    def __init__(self):
        super(snn, self).__init__()
        self.threshold = 0.8
        self.slinear = nn.Sequential(
            nn.Linear(30, 128),
            neuron.LIFNode(tau=2.0, v_threshold = self.threshold,surrogate_function=surrogate.ATan()),
            nn.Linear(128, 30),
            neuron.LIFNode(tau=2.0, v_threshold = self.threshold,surrogate_function=surrogate.ATan()),
            )

    def forward(self, x, snn_threshold):
        self.threshold  = snn_threshold
        self.slinear[1].v_threshold = snn_threshold
        self.slinear[3].v_threshold = snn_threshold
        output = self.slinear(x)
        return output
