#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <math.h>
#include <cuda.h>

#define COMMENT "Histogram_GPU"
#define RGB_COMPONENT_COLOR 255

//Tamanhos dos blocos das threads
#define BLOCK_SIZE 32

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

__global__ void cudaHistogram(PPMPixel* data, int rows, int cols, float* h){
	
	//Definindo variaveis locais na funcao na GPU
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = (cols)*row + col;
    int j, k, l;	

    //Verificao para os limites das threads
	if(col < (cols) && row < (rows)){
		//Searching for the right value of the pixel
		int x = 0;
		for (j = 0; j <= 3; j++) {
			for (k = 0; k <= 3; k++) {
				for (l = 0; l <= 3; l++) {
					if (data[tid].red == j && data[tid].green == k && data[tid].blue == l) {						
						atomicAdd(&h[x], 1);
					}
					
					x++;
				}				
			}
		}
		
	}

}


void Histogram(PPMImage *image, float *h) {

	
	cudaEvent_t start, stop;
	float milliseconds = 0;
	PPMPixel *pixels_dev;
	float* h_dev;
	float n = image->y * image->x;

	//printf("%d, %d\n", rows, cols );
	int i;
	for (i = 0; i < n; i++) {
		image->data[i].red = floor((image->data[i].red * 4) / 256);
		image->data[i].blue = floor((image->data[i].blue * 4) / 256);
		image->data[i].green = floor((image->data[i].green * 4) / 256);
	}
	//Processo para calcular o tempo de alocar memoria na GPU
	cudaEventCreate(&start);
  	cudaEventCreate(&stop);
	cudaEventRecord(start);  	
	cudaMalloc(&pixels_dev, sizeof(PPMPixel)*image->x*image->y);
	cudaMalloc(&h_dev, sizeof(float)*64);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	//printf("Alocar Memoria = %f\n",milliseconds/1000);


	//Calular o tempo de copiar dados da CPU para a GPU
	cudaEventCreate(&start);
  	cudaEventCreate(&stop);
	cudaEventRecord(start);  	
	cudaMemcpy(pixels_dev, image->data, image->x*image->y*sizeof(PPMPixel), cudaMemcpyHostToDevice);
	cudaMemcpy(h_dev, h, 64*sizeof(float), cudaMemcpyHostToDevice);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	//printf("\nOffload do buffer = %f\n",milliseconds/1000);


    dim3 blocks(1,1,1);
    //variavel para threadsPerBlock e o tamanho do block para cada dimensao 2D
    dim3 threadsPerBlock(BLOCK_SIZE,BLOCK_SIZE,1);

    //define a quantidade de blocos por dimensao/BLOCK_SIZE. se dimensao < block_size, entao define como 1 block
	blocks.x=((image->y/BLOCK_SIZE) + (((image->y)%BLOCK_SIZE)==0?0:1));
	blocks.y=((image->x/BLOCK_SIZE) + (((image->x)%BLOCK_SIZE)==0?0:1));

   	cudaEventCreate(&start);
  	cudaEventCreate(&stop);
	cudaEventRecord(start);  	
	cudaHistogram<<<blocks, threadsPerBlock>>> (pixels_dev, image->x, image->y, h_dev);
	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	//printf("\nTempo de kernel = %f\n",milliseconds/1000);
    

	//GPU para CPU
	cudaEventCreate(&start);
  	cudaEventCreate(&stop);
	cudaEventRecord(start);  	
 	cudaMemcpy(h, h_dev, 64*sizeof(float), cudaMemcpyDeviceToHost);	
 	cudaEventRecord(stop);
	cudaEventSynchronize(stop);
	cudaEventElapsedTime(&milliseconds, start, stop);
	//printf("\nTempo de offload para receber = %f\n",milliseconds/1001);
  
  	cudaFree(h_dev);
}

int main(int argc, char *argv[]) {

	if( argc != 2 ) {
		printf("Too many or no one arguments supplied.\n");
	}

	double t_start, t_end;
	int i;
	char *filename = argv[1]; //Recebendo o arquivo!;
	
	//scanf("%s", filename);
	PPMImage *image = readPPM(filename);
	
	float n = image->y * image->x;
	float *h = (float*)malloc(sizeof(float) * 64);

	//Inicializar h
	for(i=0; i < 64; i++) h[i] = 0.0;

	t_start = rtclock();
	Histogram(image, h);
	t_end = rtclock();

	for (i = 0; i < 64; i++){
		printf("%0.3f ", h[i]/n);
	}
	printf("\n");
	fprintf(stdout, "\n%0.6lfs\n", t_end - t_start);  
	free(h);
}
