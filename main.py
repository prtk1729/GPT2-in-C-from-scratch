import tiktoken
import torch
import safetensors

def encode_decode_file(filename):
    f = open(filename, "r")

    # need to make it know the encoding method name. Why?
    # Once, we load weights, we need to specify the enc mech
    # So, that we can go to and from between encode and decode
    enc = tiktoken.encoding_for_model("gpt2")
    tokens = enc.encode( f.read() )
    print( tokens[:3] ) # tokenised as numbers
    tokens_dec = enc.decode( tokens ) # list -> string
    print( tokens_dec[:3] ) # Works!
    return


def read_safetensor(path):
    fp = open(path, "rb") # Trap

    # deserialize this using safetensor pkg
    x = safetensors.deserialize(fp.read())

    # read and verify the safetnsor
    print( type(x) ) # <class 'list'>
    # print( x[0] ) # ("tensor_name", "actual_data_byte_format")

    weight_map = dict()
    for safetensor in x:
        name = safetensor[0]
        shape = safetensor[1]["shape"]
        data = safetensor[1]["data"]
        dtype = safetensor[1]["dtype"]

        # convert
        tensor = torch.frombuffer(buffer=data, dtype=torch.float32)
        # returns 1D tensor
        tensor = tensor.reshape(shape)
        print( name, shape, tensor.shape, dtype )

        weight_map[name] = tensor
        return weight_map





if __name__ == "__main__":
    filename = "/home/gpt2/GPT2-in-C-from-scratch/assets/tiny.txt"
    # encode_decode_file(filename) # Works!

    safetensors_path = "/home/model.safetensors"
    weight_map = read_safetensor(safetensors_path)