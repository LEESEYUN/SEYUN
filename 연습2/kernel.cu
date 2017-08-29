
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <cuda.h>
#include <cuda_runtime_api.h>
#include <device_functions.h>

#include <stdio.h>

#define MAX 33*1024
const int threadPerblock = 256;
const int gridPerblock = (MAX + threadPerblock - 1) / threadPerblock;



__global__ void multi(float*a, float*b, float*c){
	__shared__ float cache[threadPerblock];
	
	int tid = threadIdx.x + blockIdx.x*blockDim.x;
	int cacheIndex = threadIdx.x;
	float temp = 0;
	while (tid < MAX){
		temp = b[tid] * a[tid];
		tid += gridDim.x*blockDim.x;
	}
	cache[cacheIndex] = temp;
 

	__syncthreads();

	int i = blockDim.x / 2;
	while (i != 0){
		if (cacheIndex < i)
			cache[cacheIndex] += cache[cacheIndex + i];
		__syncthreads();
		i /= 2;
	}
	if (cacheIndex == 0)
		c[blockIdx.x] = cache[0];

}



int main()
{
	float a[MAX],b[MAX],c[gridPerblock];
	float* dev_a, *dev_b, *dev_c;


	for (int i = 0; i < MAX; i++){
		a[i] = i;
		b[i] = -i;
	}


	cudaMalloc((void**)&dev_a, sizeof(float)*MAX);
	cudaMalloc((void**)&dev_b, sizeof(float)*MAX);
	cudaMalloc((void**)&dev_c, sizeof(float)*gridPerblock);

	cudaMemcpy(dev_a, a, sizeof(float)*MAX, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, sizeof(float)*MAX, cudaMemcpyHostToDevice);

	multi << <threadPerblock, threadPerblock >> >(dev_a, dev_b, dev_c);

	cudaMemcpy(c, dev_c, sizeof(float)*gridPerblock, cudaMemcpyDeviceToHost);

	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);
	
	float temp=0;
	for (int i = 0; i < gridPerblock; i++){
		//temp += c[i];
		printf("%f\n", c[i]);
		/*
		if (i % 10 == 0){
			printf("\n");
		}
		*/
	}
	//printf("%f", temp);

	
    return 0;
}

