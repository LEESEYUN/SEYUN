#include <iostream>
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"

#ifdef __INTELLISENSE__
void __syncthreads();
#endif


#define ThreadPerBlock 1024

__global__ void dot(float*a, float*b, float*c, int threadperblock, int max){
	__shared__ float cache[ThreadPerBlock];
	int tid = threadIdx.x + blockDim.x*blockIdx.x;
	float temp = 0;
	int cacheindex = threadIdx.x;
	while (tid < max){
		temp = a[tid] * b[tid];
		tid += gridDim.x*blockDim.x;
	}
	cache[cacheindex] = temp;

	__syncthreads();
	int i = blockDim.x / 2;
	while (i != 0){
		if (cacheindex < i)
			cache[cacheindex] += cache[cacheindex + i];
		__syncthreads();
		i /= 2;
	}
	if (cacheindex == 0)
		c[blockIdx.x] = cache[0];



}




extern "C" void kernel(float *a, float *b, float *c, int max, int threadperblock){
	const int BlockPerGrid = (max + threadperblock - 1) / threadperblock;
	float *dev_a, *dev_b, *dev_c;
	for (int i = 0; i < max; i++){
		a[i] = (float)i;
		b[i] = (float)i*i;
	}

	cudaMalloc((void**)&dev_a, sizeof(float)*max);
	cudaMalloc((void**)&dev_b, sizeof(float)*max);
	cudaMalloc((void**)&dev_c, sizeof(float)*BlockPerGrid);

	cudaMemcpy(dev_a, a, sizeof(float)*max, cudaMemcpyHostToDevice);
	cudaMemcpy(dev_b, b, sizeof(float)*max, cudaMemcpyHostToDevice);

	dot<< <BlockPerGrid, threadperblock >> >(dev_a, dev_b, dev_c,threadperblock,max);

	cudaMemcpy(c, dev_c, sizeof(float)*BlockPerGrid, cudaMemcpyDeviceToHost);
	cudaFree(dev_a);
	cudaFree(dev_b);
	cudaFree(dev_c);

}


