#### Embedding is the first operation (Related to `unembedding layer i.e last layer; Hence weight-tying`)
> [!NOTE]
> - While working with learning map of emb => nn.Embedding()
> - Working with fixed weights F.linear() and F.embedding()
> - (seq_len) -> Get weight_embs for each  of these input tokens search in emb_dict
> - No need to transpose, it automatically does that


##### Where to find on HF and viz
![](../../images/embedding.png)
- wpe.weight: positional weights which are frozen after training (for indices 0... 1023)
- wte.weight: Tensor(input) weights which are frozen after training (for token_indices 0... seq_len-1)


##### Code Snippets and Funda

```python
    # No need to transpose, it automatically does that
    wte_out = F.embedding( input=x, \ # Ensure, input is torch.tensor
                           weight = params["wte.weight"], # No need to transpose
                            ) # implicitly funda: searches tokens position in dict and transforms
```

```python
def get_input_and_pe(x, params):
    # x: (seq_len, d_model) -> (64, 768)

    # NOTE: While working with learning map of emb => nn.Embedding()
    # Working with fixed weights F.linear() and F.embedding()
    # (seq_len) -> Get weight_embs for each  of these input tokens search in emb_dict
    # No need to transpose, it automatically does that
    wte_out = F.embedding( input=x, \
                           weight = params["wte.weight"],
                            )
    print( wte_out.shape ) # torch.Size([64, 768])


    # positional encoding -> wpe.weight is frozen repr of positions from [0, 1023]
    # each 0, 1, 2, .., 1023 indices is in 768 dim
    # need to first know seq_len of input
    seq_len = x.shape[0]
    pos_input = torch.arange(seq_len)
    # print(pos_input)
    # (seq_len) -> Get pos_embs for each  of these input tokens search in emb_dict
    wpe_out = F.embedding( input = pos_input, 
                          weight= params["wpe.weight"] )
    print( wpe_out.shape )
    return wte_out, wpe_out
```