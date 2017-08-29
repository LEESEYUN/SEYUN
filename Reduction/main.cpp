#include <iostream>
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"


#define MAX 33*1024
#define ThreadPerBlock 1024



extern "C" void kernel(float *a, float *b, float *c,int max,int threadperblock);

int main(){


	const int BlockPerGrid = (MAX + ThreadPerBlock - 1) / ThreadPerBlock;
	
	float a[MAX], b[MAX], c[BlockPerGrid];
	

	kernel(a, b, c,MAX,ThreadPerBlock);

	printf("CPU¿¡¼­\n");
	for (int i = 0; i < BlockPerGrid; i++){
		printf("%f\n", c[i]);

	}




	return 0;
}