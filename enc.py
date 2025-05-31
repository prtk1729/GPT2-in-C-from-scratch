import tiktoken
import struct

def get_enc_and_tokens(path):
    fp = open(path, "r")
    enc = tiktoken.encoding_for_model("gpt2")
    tokens = enc.encode( fp.read() )

    return enc, tokens


# Idea! -> create 2 things -> struct-like-layout and a byte-string
# Byte-pair-encoding is an overkill -> we can create a table that has [offset, size] as follows
# This table, we call as structs and write to a file
# struct s1{
#     u32 offest;
#     u32 size;
# };
# struct s1{
#     u32 offest;
#     u32 size;
# };
# struct s1{
#     u32 offest;
#     u32 size;
# };
# # Same as above layout of memory -> for a token [offset, size], offset is id
# "a", "!", "abc" -> [ 1, 1, 2, 1, 3, 3 ] => last guy offset is 3 and size is 3

# We also, store the byte-string in the same file


def get_strings_and_struct(enc):
    structs = [] # datastructure that holds the above in mem
    strings = b""

    for i in range(50257):
        # What's the string for ith token-d based on enc

        s = enc.decode_single_token_bytes(i)
        size = len(s)
        offset = len(strings) # id in string based on space

        structs.append(offset)
        structs.append(size)

        # strings += s Need to typecast in bytes
        strings += s
        # print( i, s, offset, size ) # 50256 b'<|endoftext|>' 320814 13 => last entry

        # print(structs) # for first string [0, 1]
        # print(strings) # bytes-string after 1st append: b'!'. Expected.

    return strings, structs


def write_in_file(strings, structs):
    fp = open("enc", "wb")

    # fp.write(structs) # Check file-szize -> TypeError: a bytes-like object is required, not 'list'
    # We convert them into bytes and then store
    for st in structs:
        # Handling bytes -> py pkg struct
        s = struct.pack("I", st) # we want to store in uint format
        fp.write(s)

    import os
    print( "After structs are written: ", os.path.getsize("enc") )

    # string is already in bytes
    fp.write(strings)
    print( "After strings are written: ", os.path.getsize("enc") )

    

if __name__ == "__main__":
    # create the encoding mechanism
    enc, tokens = get_enc_and_tokens("/home/tiny.txt")
    print( tokens[:5] ) # verified


    strings, structs = get_strings_and_struct(enc)
    # print( len(strings), len(structs) ) # 320827 100514

    # write these strings anbd structs in a file
    write_in_file(strings, structs)

    # filesize ?
    # import os
    # print( len(strings) , len(structs)*2 ) # 320827 201028
    # print( os.path.getsize("enc") )  # 722883 B 

