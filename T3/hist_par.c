#include <stdio.h>
#include <float.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/time.h>
#include <pthread.h>

// gcc -g -Wall -o app pi_s.c -lpthread -lm

/**
Tabela
        	1	2 		4		8		16
Arq1.in S	1	1.34	1.40	0.62	0.84
		E	1	0.67	0.35	0.07	0.05			
Arq2.in S	1	1.56	1.82	2.52	2.16			
		E	1	0.78	0.45	0.31	0.13			
Arq3.in S	1	1.78	2.40	2.28	2.26			
		E	1	0.89	0.60	0.28	0.14			
Percentual paralelizavel: 0.60
**/

double *val, h, min;
int n, nval, i, *vet, size;

/* funcao que calcula o minimo valor em um vetor */
double min_val(double * vet,int nval) {
	int i;
	double min;

	min = FLT_MAX;

	for(i=0;i<nval;i++) {
		if(vet[i] < min)
			min =  vet[i];
	}
	
	return min;
}

/* funcao que calcula o maximo valor em um vetor */
double max_val(double * vet, int nval) {
	int i;
	double max;

	max = FLT_MIN;

	for(i=0;i<nval;i++) {
		if(vet[i] > max)
			max =  vet[i];
	}
	
	return max;
}

/*Nova funcao count que sera passada para as threads*/
void * count(void * rank) {
	int i, j, count;
	double min_t, max_t;
	long my_rank = (long) rank;
	/*Variaveis que dao o range necessario para o que cada thread ira calcular*/
	int local_m = (n/size);
	int my_first = my_rank*local_m;
	int my_last = (my_rank+1)*local_m- 1;
	if (n%size!=0 ){
		my_last += n%size;
		my_rank = size-1;
	}

	for(j=my_first;j<=my_last;j++) {
		count = 0;
		min_t = min + j*h;
		max_t = min + (j+1)*h;
		for(i=0;i<nval;i++) {
			if( (val[i] <= max_t && val[i] > min_t) || (j == 0 && val[i] <= min_t) ) {
				count++;
			}
		}

		vet[j] = count;
	}

	return NULL;
}

int main(int argc, char * argv[]) {
	long thread;
	pthread_t* thread_handles;
	long unsigned int duracao;
	double max;
	struct timeval start, end;

	scanf("%d",&size);
	/* entrada do numero de dados */
	scanf("%d",&nval);
	/* numero de barras do histograma a serem calculadas */
	scanf("%d",&n);

	/* vetor com os dados */
	val = (double *)malloc(nval*sizeof(double));
	vet = (int *)malloc(n*sizeof(int));

	/* entrada dos dados */
	for(i=0;i<nval;i++) {
		scanf("%lf",&val[i]);
	}

	/* calcula o minimo e o maximo valores inteiros */
	min = floor(min_val(val,nval));
	max = ceil(max_val(val,nval));

	/* calcula o tamanho de cada barra */
	h = (max - min)/n;

	gettimeofday(&start, NULL);

	/* chama a funcao */
	thread_handles = malloc (size*sizeof(pthread_t));
	
	for(thread=0; thread<size; thread++){
		pthread_create(&thread_handles[thread], NULL, count, (void*) thread);
	}

	for (thread=0; thread < size; thread++){
		pthread_join(thread_handles[thread], NULL);
	}

	free(thread_handles);

	gettimeofday(&end, NULL);

	duracao = ((end.tv_sec * 1000000 + end.tv_usec) - \
	(start.tv_sec * 1000000 + start.tv_usec));

	printf("%.2lf",min);	
	for(i=1;i<=n;i++) {
		printf(" %.2lf",min + h*i);
	}
	printf("\n");

	/* imprime o histograma calculado */	
	printf("%d",vet[0]);
	for(i=1;i<n;i++) {
		printf(" %d",vet[i]);
	}
	printf("\n");
	
	
	/* imprime o tempo de duracao do calculo */
	printf("%lu\n",duracao);

	free(vet);
	free(val);

	return 0;
}
