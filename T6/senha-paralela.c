#define _GNU_SOURCE
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/time.h>
#include <pthread.h>

FILE *popen(const char *command, const char *type);
char filename[100];

int nt;
int done = 0;

double rtclock()
{
    struct timezone Tzp;
    struct timeval Tp;
    int stat;
    stat = gettimeofday (&Tp, &Tzp);
    if (stat != 0) printf("Error return from gettimeofday: %d",stat);
    return(Tp.tv_sec + Tp.tv_usec*1.0e-6);
}

void *funcao(void* rank){
	long my_rank = (long) rank;
	FILE * fp;
	char ret[200];
	char cmd[400];	
	char finalcmd[300] = "unzip -P%d -t %s 2>&1";

	int local_m = (500000/nt);
	int my_first = my_rank*local_m;
	int my_last = (my_rank+1)*(local_m-1);

	int i;
	for(i=my_first; i < my_last && !done; i++){
	sprintf((char*)&cmd, finalcmd, i, filename);
	//printf("Comando a ser executado: %s \n", cmd); 

    fp = popen(cmd, "r");	
	while (!feof(fp)) {
		fgets((char*)&ret, 200, fp);
		if (strcasestr(ret, "ok") != NULL) {
			printf("Senha:%d\n", i);
			//flag que faz com que as threads encerrem quando alguma acha a senha
			done = 1;
			i = 500000;			
			break;
		}
	}
	pclose(fp);
  }
  return NULL;
}

int main ()
{
	long thread;
   	pthread_t* thread_handles;
	thread_handles = malloc (nt*sizeof(pthread_t));

	free(thread_handles);


   scanf("%d", &nt);
   scanf("%s", filename);
   double t_start, t_end;

   t_start = rtclock();

   for(thread=0; thread<nt; thread++){
	pthread_create(&thread_handles[thread], NULL, funcao, (void*) thread);
	}

	for (thread=0; thread < nt; thread++){
	pthread_join(thread_handles[thread], NULL);
	}

  	t_end = rtclock();
 
  	fprintf(stdout, "%0.6lf\n", t_end - t_start);  
}
