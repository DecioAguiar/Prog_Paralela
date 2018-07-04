#include<iostream>
#include <stdlib.h>

using namespace std;

int main(int argc, char const *argv[]){
	
	int linhas = atoi(argv[1]);
	int colunas = atoi(argv[2]);

	for(int i = 0; i<linhas; i++){
		for(int j=0; j<colunas; j++){
		cout<< i << " " << j << " " << rand()%100 <<endl;
		}
	}
}
