import tiktoken
import torch
import safetensors
import torch.nn as nn
import torch.nn.functional as F
import math

#############################
# CONSTANTS 
d_model = 768
seq_len = 64
num_layers = 12
head_dim = 64
n_heads = 12
#############################

def encode_decode_file(filename):
    f = open(filename, "r")

    # need to make it know the encoding method name. Why?
    # Once, we load weights, we need to specify the enc mech
    # So, that we can go to and from between encode and decode
    enc = tiktoken.encoding_for_model("gpt2")
    tokens = enc.encode( f.read() )
    # print( tokens[:3] ) # tokenised as numbers
    tokens_dec = enc.decode( tokens ) # list -> string
    # print( tokens_dec[:3] ) # Works!
    return enc, tokens


def read_safetensor(path):
    fp = open(path, "rb") # Trap

    # deserialize this using safetensor pkg
    x = safetensors.deserialize(fp.read())

    # read and verify the safetnsor
    # print( type(x) ) # <class 'list'>
    # print( x[0] ) # ("tensor_name", "actual_data_byte_format")

    params = dict()
    for safetensor in x:
        name = safetensor[0]
        shape = safetensor[1]["shape"]
        data = safetensor[1]["data"]
        dtype = safetensor[1]["dtype"]

        # convert
        tensor = torch.frombuffer(buffer=data, dtype=torch.float32)
        # returns 1D tensor
        tensor = tensor.reshape(shape)
        # print( name, shape, tensor.shape, dtype )

        params[name] = tensor
    return params


def get_input_and_pe(x, params):
    # x: (seq_len, d_model) -> (64, 768)

    # NOTE: While working with learning map of emb => nn.Embedding()
    # Working with fixed weights F.linear() and F.embedding()
    # (seq_len) -> Get weight_embs for each  of these input tokens search in emb_dict
    # No need to transpose, it automatically does that
    wte_out = F.embedding( input=x, \
                           weight = params["wte.weight"],
                            )
    print( "wte_out Shape: ", wte_out.shape ) # torch.Size([64, 768])


    # positional encoding -> wpe.weight is frozen repr of positions from [0, 1023]
    # each 0, 1, 2, .., 1023 indices is in 768 dim
    # need to first know seq_len of input
    seq_len = x.shape[0]
    pos_input = torch.arange(seq_len)
    # print(pos_input)
    # (seq_len) -> Get pos_embs for each  of these input tokens search in emb_dict
    wpe_out = F.embedding( input = pos_input, 
                          weight= params["wpe.weight"] )
    print( "wpe_out Shape: ", wpe_out.shape ) # torch.Size([64, 768])


    embedding_out = wte_out + wpe_out
    return embedding_out


def get_mask(seq_len):
    # Need to create a boolean matrix s.t 1 means mask
    # for seq_id = i => mask from {i+1, i+2, ... seq_len-1}
    # triu, upper triangular, but this makes diag = 1
    # make use of dist
    mask = torch.ones((seq_len, seq_len), dtype = bool)
    mask = torch.triu( mask, diagonal = 1 ) # Starts 1 level from actual diagonal 
    # print( mask ) # But, need to be boolean for mask_filled to work
    # print( mask )
    return mask

def encoder_layer(params, layer_idx, layer_inp):
    #############################
    # ln_1
    # (seq_len, d_model) -> (seq_len, d_model)
    ln_1_out = F.layer_norm( \
                            input = layer_inp, 
                            normalized_shape = [d_model], # layer_norm on feats i.e aggregation of feats
                            weight = params[f"h.{layer_idx}.ln_1.weight"],
                            bias = params[f"h.{layer_idx}.ln_1.bias"],
                            eps = 1e-5
                            ) # recall xi = (xi - mu) / sqrt( var**2 ) + eps
    # print( "Shape: ", ln_1_out.shape ) # (seq_len, d_model)
    #############################

    #############################
    # attention

    # (input) @ (q_proj, k_proj, v_proj) -> (q, k, v); This is how we get the q, k, v tensors
    # h.0.attn.c_attn.weight	[768, 2 304], This has all 3 q, k, v Hence, 768*3
    # q_proj.shape = [d_model, d_model] i.e [768, 768], so for others
    # (seq_len, d_model) @ (d_model, 3*d_model) -> (seq_len, 3*d_model)
        # (seq_len, d_model) @ (d_model, 3*d_model) => Linear implictly switches the dimensions
        # So, we transpose them explicitly: Trap
    attn_c_attn_out = F.linear( input = ln_1_out,
                                weight = params[f"h.{layer_idx}.attn.c_attn.weight"].transpose(-1, -2),
                                bias = params[f"h.{layer_idx}.attn.c_attn.bias"],
                               )
    # print( attn_c_attn_out.shape ) # (seq_len, 3*d_model)

    # attn_c_attn_out has q, k, v combined; split them
    q, k, v = attn_c_attn_out.split( split_size=d_model, dim=-1 )
    # print( q.shape ) # (seq_len, d_model)


    # Split each into head and store the results
    out_tensor = torch.zeros(seq_len, d_model)
    assert d_model%head_dim == 0, "d_model isn't divisible by head_dim"
    for head_idx in range(12): # NOTE: We aren't leveraging gpu here, this is done sequentially
        # operate on which part?
        # (seq_len, d_model) @ (d_model, seq_len) -> (seq_len, seq_len)
        a = torch.matmul( q[ :, head_idx*head_dim : (head_idx+1)*head_dim ], 
                      k[ :, head_idx*head_dim : (head_idx+1)*head_dim ].transpose(-1, -2) )
        # print( "a's shape: ", a.shape) 
        scale = math.sqrt( head_dim )
        a /= scale

        # For this head, how much each token attends each other token is known
        # For a given seq_position "i", we should not show the future ones
        # mask them => Create a bool mask that tells 1: mask it ow, don't
        mask = get_mask(seq_len)
        a = torch.masked_fill( input=a, mask=mask, value = -torch.inf)

        # Here, we want to softmax on each row, to normalise the weights
        # (seq_len, seq_len)
        s = torch.softmax( a, dim=-1 )

        # Weighted average of values to get the contextualised emb for each token of seq, by this given head
        # (seq_le, seq_len) @ (seq_len, head_dim) -> (seq_len, head_dim) = ( 64, 768/12 )
        attn_scores = torch.matmul(s, v[ :, head_idx*head_dim : (head_idx+1)*head_dim ])
        # print(attn_scores.shape)

        # Store this head's result at correct slice
        out_tensor[ :, head_idx*head_dim : (head_idx+1)*head_dim ] = attn_scores # reshape that we do in gpu

    # print(out_tensor.shape)
    # Finally we aggregate the indepet patterns recognised by each head
    # h.0.attn.c_proj.weight # (768, 768)

    # (seq_len, d_model ) @ (d_model, d_model) -> (seq_len, d_model)
    attn_c_proj_out = F.linear( input = out_tensor, \
                               weight = params[f"h.{layer_idx}.attn.c_proj.weight"].transpose(-1, -2),
                               bias = params[f"h.{layer_idx}.attn.c_proj.bias"],
                               )
    # print( attn_c_proj_out.shape )
    #############################

    #############################
    # residual
    # (seqw_len, d_model) + (seq_len, d_model)
    res1_out = layer_inp + attn_c_proj_out
    #############################

    #############################   
    # ln_2 : h.0.ln_2.weight
     # (seq_len, d_model) -> (seq_len, d_model)
    ln_2_out = F.layer_norm( \
                            input = res1_out, 
                            normalized_shape = [d_model], # layer_norm on feats i.e aggregation of feats
                            weight = params[f"h.{layer_idx}.ln_2.weight"],
                            bias = params[f"h.{layer_idx}.ln_2.bias"],
                            eps = 1e-5
                            ) # recall xi = (xi - mu) / sqrt( var**2 ) + eps
    # print( "Shape: ", ln_2_out.shape ) # (seq_len, d_model)   
    #############################

    #############################
    # mlp
    # up_proj and then down_proj
    # h.0.mlp.c_fc.weight: [768, 3 072]
    # (seq_len, d_model) @ (d_model, 4 * d_model) -> (seq_len, 4*d_model)
    # up_proj
    mlp_c_fc_out = F.linear( input = ln_2_out, 
                             weight = params[f"h.{layer_idx}.mlp.c_fc.weight"].transpose(-1, -2), # swap dim
                             bias = params[f"h.{layer_idx}.mlp.c_fc.bias"] )
    # print(mlp_c_fc_out.shape)

    # gelu -> no change in shape
    # gelu = x * cdf(x)
    mlp_gelu_out = F.gelu(mlp_c_fc_out)

    # down_proj
    # h.0.mlp.c_proj.weight: [3 072, 768]
    # (seq_len, 4*d_model) * (4*d_model, d_model) -> (seq_len, d_model)
    mlp_c_proj_out = F.linear( input = mlp_gelu_out, 
                             weight = params[f"h.{layer_idx}.mlp.c_proj.weight"].transpose(-1, -2), # swap dim
                             bias = params[f"h.{layer_idx}.mlp.c_proj.bias"] )
    # print(mlp_c_proj_out.shape)
    #############################

    #############################
    # residual
    res2_out = res1_out + mlp_c_proj_out
    # print(res2_out.shape) # (seq_len, d_model)
    #############################

    return res2_out


def encoder(params, embedding_out):
    # ith layer:
        # input: For 0th layer embedding_out, ow out of prev_layer
    # Start small: Let's code for 1st layer
    layer_out = [] # dummy init
    for layer_i in range(num_layers):
        layer_inp = embedding_out if layer_i == 0 else layer_out
        layer_out = encoder_layer(params, layer_i, layer_inp)
    return layer_out


def get_loss(params, encoder_out, y_true):
    # recall: encoder_out -> post_layernorm -> unembedding -> celoss(x) = criterion( y_pred, y_true )

    # post_layernorm
    # ln_f.weight	[768]
    ln_f_out = F.layer_norm( input = encoder_out, \
                            normalized_shape= [d_model],
                            weight = params["ln_f.weight"],
                            bias = params["ln_f.bias"] ,
                            eps = 1e-5 
                            )

    # unembedding (weight-tying) : we use the same F.embedding layer to get back to vocab_space
    # (seq_len, d_model) @ (d_model, vocab_size) -> (seq_len, vocab_size)
    # [seq_id = 0] => [ x1 x2 .... x50256 ] => xi's are logits, max logit value is pred
    y_pred = F.linear( input= ln_f_out, 
              weight = params["wte.weight"] )
    print( "y_pred Shape: ", y_pred.shape ) # torch.Size([64, 768])

    loss = F.cross_entropy( y_pred, y_true )
    return loss, y_pred


if __name__ == "__main__":
    filename = "/home/tiny.txt"
    enc, tokens = encode_decode_file(filename) # Returns encoding mech and tokenised string of input

    # Viz next token pred task
    # x, y_true (input, gt)
    x = tokens[:64]
    x = torch.tensor( x, dtype = torch.long )
    y_true = tokens[1:65] # seq to seq model => (0) -> (gt=1) ; (0, 1 contextualised) -> (gt=2) ; ... 
    y_true = torch.tensor( y_true, dtype = torch.long )


    safetensors_path = "/home/model.safetensors"
    params = read_safetensor(safetensors_path)

    # embedding layer
    embedding_out = get_input_and_pe(x, params)

    # encoder
    encoder_out = encoder(params, embedding_out)
    print( encoder_out.shape )

    # recall: encoder_out -> post_layernorm -> unembedding -> celoss(x) = criterion( y_pred, y_true )
    loss, y_pred = get_loss(params, encoder_out, y_true)
    print("Loss is: ", loss.item()) 

    print( enc.decode(list(x)) )
    print( "Last token: ", enc.decode([list(x)[-1]]) ) # last token
    print( "gt for next actual token: ", enc.decode([list(y_true)[-1]]) ) # last token

    # predicted token
    last_pred_token_idx = torch.argmax( y_pred[-1, :], dim = -1 )
    print( last_pred_token_idx )
    print( "next pred token: ", enc.decode( [last_pred_token_idx] ) ) # last token




