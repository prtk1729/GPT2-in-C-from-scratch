#### File I/O basics

```C
#include<stdio.h>
#include<assert.h>

int main(){

    {
        /* file pointer opening and closing best prac */
        FILE *fp = fopen("enc", "r");
        // No null ptr is returned check
        assert(fp);
        fclose(fp); // fclose(stream) : stream here is pointer to an open file, fclose flusehes this when closed
    }
    printf("test\n");
}
```


#### Idea (To create a structure called `decoder` that can assoc <offset, size, string> can be loaded unto memory in a single chunk)
