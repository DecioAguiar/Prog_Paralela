#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

void producer_consumer(int *buffer, int size, int *vec, int n, int nt) {
	int i, j;
	long long unsigned int k = 0, sum = 0;

	#pragma omp parallel num_threads(nt) private(i,j) reduction(+: sum)
	for(i=0;i<n;i++) {
		if(i % 2 == 0) {	// PRODUTOR
			#pragma omp for
			for(j=0;j<size;j++) {
				buffer[j] = vec[i] + j*vec[i+1];
			}
		}
		else {	// CONSUMIDOR
			#pragma omp for
			for(j=0;j<size;j++) {
				sum += buffer[j];
			}
		}
	}
	printf("%llu\n",sum);
}

int main(int argc, char * argv[]) {
	double start, end;
	float d;
	int i, n, size, nt;
	int *buff;
	int *vec;

	scanf("%d %d %d",&nt,&n,&size);
	
	buff = (int *)malloc(size*sizeof(int));
	vec = (int *)malloc(n*sizeof(int));

	for(i=0;i<n;i++)
		scanf("%d",&vec[i]);
	
	start = omp_get_wtime();
	producer_consumer(buff, size, vec, n, nt);
	end = omp_get_wtime();

	printf("%lf\n",end-start);

	free(buff);
	free(vec);

	return 0;
}



/*Testes*/ 