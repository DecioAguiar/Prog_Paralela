#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

// gcc -g -Wall -fopenmp -o app count_sort_serial.c 

void count_sort_serial(double a[], int n) {
	int i, j, count;
	double *temp;

	temp = (double *)malloc(n*sizeof(double));

	for (i = 0; i < n; i++) {
		count = 0;
		for (j = 0; j < n; j++)
			if (a[j] < a[i])
				count++;
			else if (a[j] == a[i] && j < i)
				count++;
		temp[count] = a[i];
	}


	memcpy(a, temp, n*sizeof(double));
	free(temp);

}

int main(){	
	
	int t; // numero de threads
	int n; // tamanho do vetor
	
	scanf ("%d",&t);
	scanf ("%d",&n);
	
	double a[n+1];
	
	for (int i = 0; i < n; i++){
		float x;
		scanf("%f",&x);
		a[i] = x;
	}
	
	double start, end, duracao;
	
	start = omp_get_wtime();
	
	#pragma omp parallel num_threads(t)
	count_sort_serial(a, n);
	
	end = omp_get_wtime();

	duracao = end - start;

	for (int i = 0; i<n; i++){
		printf("%.2f ", a[i]);	
	}
	printf("\n");
	printf("%f", duracao);

	return 0;

}