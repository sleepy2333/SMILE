import sys
import copy
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

def gelu(x):
    """Implementation of the gelu activation function.
        For information: OpenAI GPT's gelu is slightly different (and gives slightly different results):
        0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))
        Also see https://arxiv.org/abs/1606.08415
    """
    return x * 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": gelu, "relu": torch.nn.functional.relu, "swish": swish}

class PositionalEncoding(nn.Module):
    r"""Inject some information about the relative or absolute position of the tokens in the sequence.
        The positional encodings have the same dimension as the embeddings, so that the two can be summed.
        Here, we use sine and cosine functions of different frequencies.
    .. math:
        \text{PosEncoder}(pos, 2i) = sin(pos/10000^(2i/d_model))
        \text{PosEncoder}(pos, 2i+1) = cos(pos/10000^(2i/d_model))
        \text{where pos is the word position and i is the embed idx)
    Args:
        d_model: the embed dim (required).
        dropout: the dropout value (default=0.1).
        max_len: the max. length of the incoming sequence (default=5000).
    Examples:
        >>> pos_encoder = PositionalEncoding(d_model)
    """

    def __init__(self, d_model, dropout=0.0, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        d_origin = d_model
        if d_model%2 != 0:    
            d_model += 1
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        if d_origin%2 != 0:
            pe = pe[:,:-1]
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        r"""Inputs of forward function
        Args:
            x: the sequence fed to the positional encoder model (required).
        Shape:
            x: [batch size, sequence length, embed dim]
            output: [batch size, sequence length, embed dim]
        Examples:
            >>> output = pos_encoder(x)
        """
        #inputs 3D [N,L,D]
        x = x + self.pe[:, :x.size(1)]
        
        return self.dropout(x)


class Config(object):
    """Configuration class to store the configuration of a `BertModel`.
    """
    def __init__(self,
                 hidden_size=128,
                 num_hidden_layers=3,
                 num_attention_heads=4,
                 intermediate_size=512,
                 hidden_act="relu",
                 hidden_dropout_prob=0.1,
                 attention_probs_dropout_prob=0,
                 max_position_embeddings=2048,
                 type_vocab_size=32,
                 #initializer_range=0.02,
                 #kernel_size=3
                 ):
        """Constructs BertConfig.

        Args:
            hidden_size: Size of the encoder layers and the pooler layer.
            num_hidden_layers: Number of hidden layers in the Transformer encoder.
            num_attention_heads: Number of attention heads for each attention layer in
                the Transformer encoder.
            intermediate_size: The size of the "intermediate" (i.e., feed-forward)
                layer in the Transformer encoder.
            hidden_act: The non-linear activation function (function or string) in the
                encoder and pooler. If string, "gelu", "relu" and "swish" are supported.
            hidden_dropout_prob: The dropout probabilitiy for all fully connected
                layers in the embeddings, encoder, and pooler.
            attention_probs_dropout_prob: The dropout ratio for the attention
                probabilities.
            max_position_embeddings: The maximum sequence length that this model might
                ever be used with. Typically set this to something large just in case
                (e.g., 512 or 1024 or 2048).
            type_vocab_size: The vocabulary size of the `token_type_ids` passed into
                `BertModel`.
            initializer_range: The sttdev of the truncated_normal_initializer for
                initializing all weight matrices.
        """
        self.hidden_size = hidden_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.hidden_act = hidden_act
        self.intermediate_size = intermediate_size
        self.hidden_dropout_prob = hidden_dropout_prob
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings
        self.type_vocab_size = type_vocab_size
        #self.initializer_range = initializer_range
        #self.kernel_size = kernel_size


class LayerNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-12):
        """Construct a layernorm module in the TF style (epsilon inside the square root).
        """
        super(LayerNorm, self).__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.bias = nn.Parameter(torch.zeros(hidden_size))
        self.variance_epsilon = eps

    def forward(self, x):
        u = x.mean(-1, keepdim=True)
        s = (x - u).pow(2).mean(-1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.variance_epsilon)
        return self.weight * x + self.bias


class Embeddings(nn.Module):
    """Construct the embeddings from word, position and token_type embeddings.
    """
    def __init__(self, config):
        super(Embeddings, self).__init__()
        self.position_embeddings = PositionalEncoding(max_len = config.max_position_embeddings, d_model = config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size, padding_idx=0)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, inputs, token_type_ids=None):
        #inputs 3D [N,L,D]
        if token_type_ids is None:
            token_type_ids = torch.zeros(*inputs.shape[:-1]).to(dtype=torch.long,device=inputs.device)

        position_embeddings   = self.position_embeddings(inputs)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        
        embeddings = inputs + position_embeddings + token_type_embeddings#
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings

#
class CrossAttention(nn.Module):
    def __init__(self, config):
        super(CrossAttention, self).__init__()
        if config.hidden_size % config.num_attention_heads != 0:
            raise ValueError(
                "The hidden size (%d) is not a multiple of the number of attention "
                "heads (%d)" % (config.hidden_size, config.num_attention_heads))
        self.num_attention_heads = config.num_attention_heads
        self.attention_head_size = int(config.hidden_size / config.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        #'''
        self.query = nn.Linear(config.hidden_size, self.all_head_size)
        self.key   = nn.Linear(config.hidden_size, self.all_head_size)
        self.value = nn.Linear(config.hidden_size, self.all_head_size)
        '''
        self.query = nn.Conv1d(config.hidden_size, self.all_head_size, config.kernel_size, padding=1)
        self.key   = nn.Conv1d(config.hidden_size, self.all_head_size, config.kernel_size, padding=1)
        self.value = nn.Conv1d(config.hidden_size, self.all_head_size, config.kernel_size, padding=1)
        #'''


        self.dropout = nn.Dropout(config.attention_probs_dropout_prob)
    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, queries, keys, values, attention_mask):
        #input shape: N*L*D
        #'''
        mixed_query_layer = self.query(queries)
        mixed_key_layer   = self.key(keys)
        mixed_value_layer = self.value(values)
        '''
        queries = queries.transpose(-1,-2)
        keys    = keys.transpose(-1,-2)
        values  = values.transpose(-1,-2)
        mixed_query_layer = self.query(queries).transpose(-1,-2)
        mixed_key_layer   = self.key(keys).transpose(-1,-2)
        mixed_value_layer = self.value(values).transpose(-1,-2)
        #'''

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask        
        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)#[N, Head, Seq_L_q, Seq_L_k]
        

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)#outputs 3D [N,L,D]
        return (context_layer, attention_probs)


class MultiheadOutput(nn.Module):
    def __init__(self, config):
        super(MultiheadOutput, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.LayerNorm = LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

    def forward(self, hidden_states, input_tensor):
        #multihead attention: WO
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        #add & norm
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

#
class AttentionLayer(nn.Module):
    def __init__(self, config):
        super(AttentionLayer, self).__init__()
        self.crattn = CrossAttention(config)
        self.output = MultiheadOutput(config)

    def forward(self, queries, keys, values, attention_mask):
        crossattn_output = self.crattn(queries, keys, values, attention_mask)
        attention_output = self.output(crossattn_output[0], queries)
        return (attention_output, crossattn_output[1]) #([N, L, D], [N, Head, Seq_L_q, Seq_L_k])


class Intermediate(nn.Module):
    def __init__(self, config):
        super(Intermediate, self).__init__()
        self.dense = nn.Linear(config.hidden_size, config.intermediate_size)
        if isinstance(config.hidden_act, str) or (sys.version_info[0] == 2 and isinstance(config.hidden_act, str)):
            self.intermediate_act_fn = ACT2FN[config.hidden_act]
        else:
            self.intermediate_act_fn = config.hidden_act

    def forward(self, hidden_states):
        #max(0,xW1+b1)
        hidden_states = self.dense(hidden_states)
        hidden_states = self.intermediate_act_fn(hidden_states)
        return hidden_states


class OutputLayer(nn.Module):
    def __init__(self, config):
        super(OutputLayer, self).__init__()
        self.dense = nn.Linear(config.intermediate_size, config.hidden_size)
        self.LayerNorm = LayerNorm(config.hidden_size, eps=1e-12)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        
    def forward(self, hidden_states, input_tensor):
        #W2 b2
        hidden_states = self.dense(hidden_states)
        hidden_states = self.dropout(hidden_states)
        hidden_states = self.LayerNorm(hidden_states + input_tensor)
        return hidden_states

#
class BasicLayer(nn.Module):
    def __init__(self, config):
        super(BasicLayer, self).__init__()
        self.attention = AttentionLayer(config)
        self.intermediate = Intermediate(config)
        self.output = OutputLayer(config)

    def forward(self, queries, keys, values, attention_mask):
        attention_output = self.attention(queries, keys, values, attention_mask)
        intermediate_output = self.intermediate(attention_output[0])
        layer_output = self.output(intermediate_output, attention_output[0])
        return (layer_output, attention_output[1]) #([N, L, D], [N, Head, Seq_L_q, Seq_L_k])


class CrossLayer(nn.Module):
    def __init__(self, config):
        super(CrossLayer, self).__init__()
        self.AasQ = AttentionLayer(config)
        self.BasQ = AttentionLayer(config)
        self.intermediateA = Intermediate(config)
        self.intermediateB = Intermediate(config)
        self.outputA = OutputLayer(config)
        self.outputB = OutputLayer(config)
 
    def forward(self, A, B, attention_maskA, attention_maskB):
        '''
        attention_maskA is used when A is treated as a key-value sequence
        attention_maskb is used when B is treated as a key value sequence
        '''
        AasQ_output = self.AasQ(A, B, B, attention_maskB)
        BasQ_output = self.BasQ(B, A, A, attention_maskA)
        Aintermediate_output = self.intermediateA(AasQ_output[0])
        Bintermediate_output = self.intermediateB(BasQ_output[0])
        Alayer_output = self.outputA(Aintermediate_output, AasQ_output[0])
        Blayer_output = self.outputB(Bintermediate_output, BasQ_output[0])
        return (Alayer_output, Blayer_output, AasQ_output[1], BasQ_output[1])


#
class TransformerEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        ###################################
        self.embeddings = Embeddings(config)
        ###################################
        layer = BasicLayer(config)#Transformer block
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])

    def forward(self, queries, keys=None, values=None, attention_mask=None, output_all_encoded_layers=True, output_attentions=True):
        hidden_states = self.embeddings(queries)
        if keys is not None:
            keys      = self.embeddings(keys)
            values    = self.embeddings(values)
        if attention_mask is None:
            attention_mask = torch.ones(*queries.shape[:-1]) if keys is None else torch.ones(*keys.shape[:-1])
            attention_mask = attention_mask.to(device=queries.device)
        # Sizes are [batch_size, 1, 1, to_seq_length]
        extended_attention_mask = attention_mask.unsqueeze(1).unsqueeze(2)
        extended_attention_mask = extended_attention_mask.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
        extended_attention_mask = (1.0 - extended_attention_mask) * -10000.0

        #output is the last element of the list
        all_encoder_layers  = []
        all_self_attentions = [] 

        for layer_module in self.layer:
            #low-level key-value pairs as MulT
            if keys is not None:
                layer_outputs = layer_module(hidden_states, keys, values, extended_attention_mask)
            else:
                layer_outputs = layer_module(hidden_states, hidden_states, hidden_states, extended_attention_mask)

            hidden_states = layer_outputs[0]
            #######################################################################################
            #hidden_states = F.avg_pool1d(hidden_states.permute(0,2,1),kernel_size=2).permute(0,2,1)
            #######################################################################################

            if output_all_encoded_layers:
                all_encoder_layers.append(hidden_states)
            if output_attentions:
                all_self_attentions.append(layer_outputs[1])

        if not output_all_encoded_layers:
            all_encoder_layers.append(hidden_states)

        return (all_encoder_layers, all_self_attentions)



class CrossEncoder(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.embeddings = Embeddings(config)
        layer = CrossLayer(config)
        self.layer = nn.ModuleList([copy.deepcopy(layer) for _ in range(config.num_hidden_layers)])

    def forward(self, A, B, attention_maskA=None, attention_maskB=None, output_all_encoded_layers=True, output_attentions=True):
        #############################################
        a, b = self.embeddings(A), self.embeddings(B)


        #############################################

        Aall_encoder_layers, Ball_encoder_layers = [], []
        Aall_self_attentions,  Ball_self_attentions = [], []

        for layer_module in self.layer:




            attention_maskA = torch.ones(*a.shape[:-1])
            attention_maskB = torch.ones(*b.shape[:-1]) 

            attention_maskA = attention_maskA.to(device=A.device)
            attention_maskB = attention_maskB.to(device=B.device)

            extended_attention_maskA = attention_maskA.unsqueeze(1).unsqueeze(2)
            extended_attention_maskA = extended_attention_maskA.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
            extended_attention_maskA = (1.0 - extended_attention_maskA) * -10000.0
            extended_attention_maskB = attention_maskB.unsqueeze(1).unsqueeze(2)
            extended_attention_maskB = extended_attention_maskB.to(dtype=next(self.parameters()).dtype) # fp16 compatibility
            extended_attention_maskB = (1.0 - extended_attention_maskB) * -10000.0 



            layer_outputs = layer_module(a, b, extended_attention_maskA, extended_attention_maskB)
            a, b = layer_outputs[0], layer_outputs[1]
            ###############################################################
            #a = F.avg_pool1d(a.permute(0,2,1),kernel_size=2).permute(0,2,1)
            #b = F.avg_pool1d(b.permute(0,2,1),kernel_size=2).permute(0,2,1)
            ###############################################################

            if output_all_encoded_layers:
                Aall_encoder_layers.append(a)
                Ball_encoder_layers.append(b)

            if output_attentions:
                Aall_self_attentions.append(layer_outputs[2])
                Ball_self_attentions.append(layer_outputs[3])

        if not output_all_encoded_layers:
            Aall_encoder_layers.append(a)
            Ball_encoder_layers.append(b)

        return (Aall_encoder_layers, Ball_encoder_layers, Aall_self_attentions, Ball_self_attentions)


from decoder import DecoderConfig, DecoderModel
import numpy as np
class Perceiver(nn.Module):
    def __init__(self):
        super().__init__()
        self.orig_d_l, self.orig_d_a, self.orig_d_v = 300, 74, 35
        self.d_l, self.d_a, self.d_v = 30, 30, 30
        self.mapping    = nn.Embedding(num_embeddings=512, embedding_dim=30, padding_idx=0)
        self.embeddings = Embeddings(Config(hidden_size=30, intermediate_size=120))
        crosslayer = BasicLayer(Config(hidden_size=30, intermediate_size=120, num_attention_heads=5))
        selflayer  = BasicLayer(Config(hidden_size=30,intermediate_size=120, num_attention_heads=5))
        self.num_latent = 64
        self.num_layers = 12
        self.cross = nn.ModuleList([copy.deepcopy(crosslayer) for _ in range(self.num_layers)])
        self.selfa = nn.ModuleList([copy.deepcopy(selflayer)  for _ in range(self.num_layers)])
        self.proj_l = nn.Conv1d(self.orig_d_l, self.d_l, kernel_size=1, padding=0, bias=False)
        self.proj_a = nn.Conv1d(self.orig_d_a, self.d_a, kernel_size=1, padding=0, bias=False)
        self.proj_v = nn.Conv1d(self.orig_d_v, self.d_v, kernel_size=1, padding=0, bias=False)
        self.embed_dropout = 0.25
        self.nclasses = 6

        self.decoder_config = self.get_decoder_config()
        self.decoder        = DecoderModel(self.decoder_config)
        self.classifier = EmotionClassifier(30, 1)
    
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

    def forward(self, x, y, z):
        bsz = x.shape[0]
        x_l = F.dropout(x.transpose(1, 2), p=self.embed_dropout, training=self.training)
        x_a = y.transpose(1, 2)
        x_v = z.transpose(1, 2)
       
        # Project the textual/visual/audio features
        proj_x_l = x_l if self.orig_d_l == self.d_l else self.proj_l(x_l)
        proj_x_a = x_a if self.orig_d_a == self.d_a else self.proj_a(x_a)
        proj_x_v = x_v if self.orig_d_v == self.d_v else self.proj_v(x_v)
        proj_x_a = proj_x_a.permute(0, 2, 1)
        proj_x_v = proj_x_v.permute(0, 2, 1)
        proj_x_l = proj_x_l.permute(0, 2, 1)



        raw      = torch.cat([proj_x_l, proj_x_a, proj_x_v], dim=1) 
        latents  = torch.tensor([i for i in range(1, self.num_latent+1)],dtype=torch.int).unsqueeze(0).to(device=x.device)
        latents  = self.mapping(latents) #[1,L,D]
        type_idx = torch.tensor([1 for _ in range(x.shape[1])] + [2 for _ in range(y.shape[1])] + [3 for _ in range(z.shape[1])],dtype=torch.int).unsqueeze(0).to(device=x.device) 
        raw      = self.embeddings(raw, type_idx)#[N,L,D]
        selfa_mask = torch.ones(*latents.shape[:-1]).unsqueeze(1).unsqueeze(1).to(device=x.device)
        cross_mask = torch.ones(*raw.shape[:-1]).unsqueeze(1).unsqueeze(1).to(device=x.device)
        for i in range(self.num_layers):
            #latents = latents.transpose(-1,-2)
            latents = self.cross[i](latents, raw, raw, cross_mask)[0]
            #latents = latents.transpose(-1,-2)
            latents = self.selfa[i](latents, latents, latents, selfa_mask)[0]

        label_input    = torch.tensor(np.arange(self.nclasses)).to(device=x.device).unsqueeze(0)
        label_masks    = torch.ones(bsz, len(label_input)).to(device=x.device)
        latent_masks   = torch.ones(latents.shape[:-1]).to(x.device)
        last_hs_proj   = self.decoder(label_input, latents, label_masks, latent_masks)
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