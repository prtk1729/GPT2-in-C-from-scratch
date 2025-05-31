### Python (And writing unbto file and making observations)


#### First we need `enc` and `tokens` at each index based on enc-mechanism
```python
def get_enc_and_tokens(path):
    fp = open(path, "r")
    enc = tiktoken.encoding_for_model("gpt2")
    tokens = enc.encode( fp.read() )

    return enc, tokens

# create the encoding mechanism
enc, tokens = get_enc_and_tokens("/home/tiny.txt")
```

#### Create a DS that can associate `decoder-table to decode offset(id) and size(in B) for a given string`
```python
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
```


#### `structs` and `byte-strings` creation and writing
> [!NOTE]
> - Before writing unto file, we need the values to be in `byte` (`alignment` and other reasons)
> - Mistake I did:-
    ```python
        s = enc.decode([i]) # Trap: pass list, There's a better way that directly converts to byte

        # strings += s Need to typecast in bytes
        strings += bytes(s, encoding = "utf-8") # This is forcing the format to be in utf-8

        # Issue: Some chars mayn't be supported
    ```
    `Solution:-`
    ```python
        s = enc.decode_single_token_bytes(i)
        strings += s
    ```

Overall function that solves this:-
```python
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
```


#### Writing unto file
```python
def write_in_file(strings, structs):
    fp = open("enc", "wb")

    # fp.write(structs) # Check file-szize -> TypeError: a bytes-like object is required, not 'list'
    # We convert them into bytes and then store
    for st in structs:
        # Handling bytes -> py pkg struct
        s = struct.pack("I", st) # we want to store in uint format
        fp.write(s)

    # string is already in bytes
    fp.write(strings)
```

#### filesize checking in python (later will be used in C-prog)
```python
    # filesize ?
    import os
    print( len(strings) + len(structs)*2 )
    print( os.path.getsize("enc") )  # 722883 B 
```