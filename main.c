#include<stdio.h>
#include<assert.h>
#include<stdint.h>
#include<stdlib.h>

#define vocab_size 50257
#define d_enc_file_size 722883

struct token{
    uint32_t offset; // memory related always whole nums so uint32_t
    uint32_t size; 
};

// create a arr of these struictures 
// holding this idx amd subsequent info
/* Also, this decoder should support efc thing the relavant string*/
struct decoder{
    struct token tokens[vocab_size]; // list of tokens
    char byte_string_mem[320827]; // b"" in python when we check len() -> this number is shown
};

int main(){

    /* allocate memory for decoder
        -> decoder's size? 
    */
    uint32_t offset = 0, decoder_offset = 0;
    offset += sizeof(struct decoder); 

    // We need this to be present in memory
    char* raw_mem = (char*)malloc( offset ); // raw mem is decoder's total mem req
    assert(raw_mem); // non-null i.e can be alloc

    // create a dec_ptr pointing to the end of raw_mem
    // This we will use along with fp, to read data
    // dec_ptr points just after raw_mem ends
    struct decoder* dec_ptr = (struct decoder*)(raw_mem + decoder_offset);

    /* file pointer opening and closing best prac */
    /*        size_t fread(void ptr[restrict .size * .nmemb],
                    size_t size, size_t nmemb,
                    FILE *restrict stream); */
    FILE *fp = fopen("enc", "r");
    // No null ptr is returned check
    assert(fp);

    // read contents of enc using fp(stream) and ptr 
    // unsigned long fsize = fread( dec_ptr, 1, sizeof(struct decoder), fp ); // rets number of itens (i.e X bytes)
    unsigned long fsize = fread( dec_ptr, 1, d_enc_file_size, fp ); // rets number of itens (i.e X bytes)
    
    printf("%ld %ld", fsize, d_enc_file_size);
    assert( fsize == d_enc_file_size) );


    // Print the decoder structure
    for(int i=0; i<vocab_size; i++){
        uint32_t size = dec_ptr->tokens[i].size;
        uint32_t offset = dec_ptr->tokens[i].offset;

        // also need to print string using *.s -> sz, string_start_ptr
        printf("%d %u %u %.*s\n", i, offset, size, size, dec_ptr->byte_string_mem + offset);
    }


    fclose(fp); // fclose(stream) : stream here is pointer to an open file, fclose flusehes this when closed    

    printf("test\n");
}
