#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define M 1024

__global__ void sumMatrixes(int* A, int* B, int* C, int n){
    int index = threadIdx.x + blockIdx.x * blockDim.x;
    if(index < n){
        C[index] = A[index] + B[index];      
    }
    
}

int main(void){
    int *A, *B, *C;
    int i, j;

    //Input
    int linhas, colunas;
    
    scanf("%d", &linhas);
    scanf("%d", &colunas);

    int N = linhas * colunas;
    int size = N * sizeof(int);

    //Alocando memória na CPU
    A = (int *)malloc(sizeof(int)*linhas*colunas);
    B = (int *)malloc(sizeof(int)*linhas*colunas);
    C = (int *)malloc(sizeof(int)*linhas*colunas);
    
    //Inicializar
    for(i = 0; i < linhas; i++){
        for(j = 0; j < colunas; j++){
            A[i*colunas+j] =  B[i*colunas+j] = i+j;
        }
    }

//tentamos alocar espaco de memoria para as matrizes na GPU, se der erro, mostramos o erro na saida padrao.
    int *_A, *_B, *_C;

    cudaMalloc((void**)&_A, size);
    cudaMalloc((void**)&_B, size);
    cudaMalloc((void**)&_C, size);

    cudaMemcpy(_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(_B, B, size, cudaMemcpyHostToDevice);

    sumMatrixes <<< (N + M-1)/ M, M >>> (_A, _B, _C, N); 

    cudaMemcpy(C, _C, size, cudaMemcpyDeviceToHost);


    //Computacao que deverá ser movida para a GPU (que no momento é executada na CPU)
    //Lembrar que é necessário usar mapeamento 2D (visto em aula) 
    // for(i=0; i < linhas; i++){
    //     for(j = 0; j < colunas; j++){
    //         C[i*colunas+j] = A[i*colunas+j] + B[i*colunas+j];
    //     }
    // }

    long long int somador=0;
    //Manter esta computação na CPU
    for(i = 0; i < linhas; i++){
        for(j = 0; j < colunas; j++){
            somador+=C[i*colunas+j];   
        }
    }
    
    printf("%lli\n", somador);

    free(A);
    free(B);
    free(C);

    cudaFree(_A);
    cudaFree(_B);
    cudaFree(_C);
}

