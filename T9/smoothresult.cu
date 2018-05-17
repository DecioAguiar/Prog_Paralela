#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#define MASK_WIDTH 5
#define TILE_WIDTH 16
#define T_Bloco (MASK_WIDTH + TILE_WIDTH -1)

#define COMMENT "Histogram_GPU"
#define RGB_COMPONENT_COLOR 255

typedef struct {
    unsigned char red, green, blue;
} PPMPixel;

typedef struct {
    int x, y;
    PPMPixel *data;
} PPMImage;

double rtclock()
{
    struct timezone Tzp;
    struct timeval Tp;
    int stat;
    stat = gettimeofday (&Tp, &Tzp);
    if (stat != 0) printf("Error return from gettimeofday: %d",stat);
    return(Tp.tv_sec + Tp.tv_usec*1.0e-6);
}


static PPMImage *readPPM(const char *filename) {
    char buff[16];
    PPMImage *img;
    FILE *fp;
    int c, rgb_comp_color;
    fp = fopen(filename, "rb");
    if (!fp) {
        fprintf(stderr, "Unable to open file '%s'\n", filename);
        exit(1);
    }

    if (!fgets(buff, sizeof(buff), fp)) {
        perror(filename);
        exit(1);
    }

    if (buff[0] != 'P' || buff[1] != '6') {
        fprintf(stderr, "Invalid image format (must be 'P6')\n");
        exit(1);
    }

    img = (PPMImage *) malloc(sizeof(PPMImage));
    if (!img) {
        fprintf(stderr, "Unable to allocate memory\n");
        exit(1);
    }

    c = getc(fp);
    while (c == '#') {
        while (getc(fp) != '\n')
            ;
        c = getc(fp);
    }

    ungetc(c, fp);
    if (fscanf(fp, "%d %d", &img->x, &img->y) != 2) {
        fprintf(stderr, "Invalid image size (error loading '%s')\n", filename);
        exit(1);
    }

    if (fscanf(fp, "%d", &rgb_comp_color) != 1) {
        fprintf(stderr, "Invalid rgb component (error loading '%s')\n",
                filename);
        exit(1);
    }

    if (rgb_comp_color != RGB_COMPONENT_COLOR) {
        fprintf(stderr, "'%s' does not have 8-bits components\n", filename);
        exit(1);
    }

    while (fgetc(fp) != '\n')
        ;
    img->data = (PPMPixel*) malloc(img->x * img->y * sizeof(PPMPixel));

    if (!img) {
        fprintf(stderr, "Unable to allocate memory\n");
        exit(1);
    }

    if (fread(img->data, 3 * img->x, img->y, fp) != img->y) {
        fprintf(stderr, "Error loading image '%s'\n", filename);
        exit(1);
    }

    fclose(fp);
    return img;
}

void writePPM(PPMImage *img) {

    fprintf(stdout, "P6\n");
    fprintf(stdout, "# %s\n", COMMENT);
    fprintf(stdout, "%d %d\n", img->x, img->y);
    fprintf(stdout, "%d\n", RGB_COMPONENT_COLOR);

    fwrite(img->data, 3 * img->x, img->y, stdout);
    fclose(stdout);
}


__global__ void cudaSmooth(PPMPixel *image, PPMPixel *image_copy, int rows, int cols){

    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int row = blockIdx.y * blockDim.y + threadIdx.y;

    int border = (int)((MASK_WIDTH - 1) / 2);
    int dimension = TILE_WIDTH + (MASK_WIDTH - 1);
    __shared__ PPMPixel shared[TILE_WIDTH + (MASK_WIDTH - 1)][TILE_WIDTH + (MASK_WIDTH - 1)];

    int size = ceil((float)(dimension * dimension) / (blockDim.x * blockDim.y));

    int block_start_col = blockIdx.x * blockDim.x;
    int block_start_row = blockIdx.y * blockDim.y;

    for(int k = 0; k < size; k++){

        int index = (threadIdx.y * blockDim.x) + threadIdx.x + (k * blockDim.x * blockDim.y);
        int x = (int)(index / dimension);
        int y = index % dimension;

        if(x < dimension && y < dimension){
            int sr = block_start_row + x - border;
            int sc = block_start_col + y - border;

            if(sr >= 0 && sc >= 0 && sc < cols && sr < rows){
                shared[x][y] = image_copy[sr * cols + sc];
            }else{
                shared[x][y].red = shared[x][y].green = shared[x][y].blue = 0;
            }
        }
    }

    __syncthreads();

    int total_red, total_blue, total_green;
    total_red = total_blue = total_green = 0;
    //Determiando a posicao da matriz
    if((row * col) <= (rows * cols)){
        for(int i = threadIdx.y; i < (threadIdx.y + MASK_WIDTH); i++){
            for(int j = threadIdx.x; j < (threadIdx.x + MASK_WIDTH); j++){
                total_red += shared[i][j].red;
                total_green += shared[i][j].green;
                total_blue += shared[i][j].blue;//if
            } //for z
        } //for y

        image[row * cols + col].red = total_red / (MASK_WIDTH*MASK_WIDTH);
        image[row * cols + col].blue = total_blue / (MASK_WIDTH*MASK_WIDTH);
        image[row * cols + col].green = total_green / (MASK_WIDTH*MASK_WIDTH);

    }


}


void Smoothing_GPU(PPMImage * image, PPMImage * image_output){
    PPMPixel *_image;
    PPMPixel *_image_copy;

    cudaMalloc(&_image, sizeof(PPMPixel)*image->x*image->y);
    cudaMalloc(&_image_copy, sizeof(PPMPixel)*image->x*image->y);

    cudaMemcpy(_image, image->data, image->x*image->y*sizeof(PPMPixel), cudaMemcpyHostToDevice);

    dim3 dimGrid(ceil((float) image->x/TILE_WIDTH), ceil((float) image->y/TILE_WIDTH ), 1);
    dim3 dimBlock(TILE_WIDTH, TILE_WIDTH);

 
    cudaSmooth<<<dimGrid, dimBlock>>> (_image_copy, _image, image->y, image->x);

    cudaMemcpy(image_output->data, _image_copy, sizeof(PPMPixel)*image->x*image->y, cudaMemcpyDeviceToHost);
    
    cudaFree(_image);
    cudaFree(_image_copy);
}


int main(int argc, char *argv[]) {

    if( argc != 2 ) {
        printf("Too many or no one arguments supplied.\n");
    }

    char *filename = argv[1]; //Recebendo o arquivo!;

    PPMImage *image = readPPM(filename);
    PPMImage *image_output = readPPM(filename);

    Smoothing_GPU(image, image_output);
    writePPM(image_output);

    free(image);
    free(image_output);
}


