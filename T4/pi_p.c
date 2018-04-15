#include<stdio.h>
#include<stdlib.h>
#include <sys/time.h>
#include <pthread.h>

// gcc -g -Wall -o app pi_s.c -lpthread -lm
unsigned int n, i;
long long unsigned int in = 0;
unsigned int semente;
int size;
pthread_mutex_t mutex;

void* monte_carlo_pi(void* rank) {
	long long unsigned int i;
	long long unsigned int local = 0;
	double x, y, d;
	long my_rank = (long) rank;

	int local_m = (n/size);
	int my_first = my_rank*local_m;
	int my_last = (my_rank+1)*local_m- 1;
	if (n%size!=0 ){
		my_last += n%size;
		my_rank = size-1;
	}

	for (i = my_first; i < my_last; i++) {
		x = ((rand_r(&semente) % 1000000)/500000.0)-1;
		y = ((rand_r(&semente) % 1000000)/500000.0)-1;
		d = ((x*x) + (y*y));
		if (d <= 1){
			local+=1;	
		} 
	}

	pthread_mutex_lock(&mutex);
	in += local;
	pthread_mutex_unlock(&mutex);

	return NULL;
}

int main(void) {
	long thread;
	pthread_t* thread_handles;
	double pi;
	long unsigned int duracao;
	struct timeval start, end;

	scanf("%d %u",&size, &n);

	srand (time(NULL));

	gettimeofday(&start, NULL);

	/* chama a funcao */
	thread_handles = malloc (size*sizeof(pthread_t));
	
	for(thread=0; thread<size; thread++){
		pthread_create(&thread_handles[thread], NULL, monte_carlo_pi, (void*) thread);
	}

	for (thread=0; thread < size; thread++){
		pthread_join(thread_handles[thread], NULL);
	}

	free(thread_handles);


//	in = monte_carlo_pi(n);
	gettimeofday(&end, NULL);

	duracao = ((end.tv_sec * 1000000 + end.tv_usec) - (start.tv_sec * 1000000 + start.tv_usec));

	pi = 4*in/((double)n);
	printf("%lf\n%lu\n",pi,duracao);

	return 0;
}
