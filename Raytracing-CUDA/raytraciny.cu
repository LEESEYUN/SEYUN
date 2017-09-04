#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../../common/book.h"
#include "../../common/cpu_bitmap.h"
#include <math.h>

#define DIM 1024
#define rnd(x) (x*rand()/RAND_MAX)
#define SPHERES 20 // 구의 숫자
#define INF 2E10f
#ifdef __INTELLISENSE__
void __syncthreads();
#endif

typedef struct Sphere{
	float r, g, b;
	float radius;  //구의 반지름
	float x, y, z; // 구의 좌표
	__device__ float hit(float ox, float oy, float *n){ //ox,oy는 ray의 좌표

		float dx = ox - x; //구의 중심과 ray를 쏜 곳 사이의 거리
		float dy = oy - y;
		if (dx*dx + dy*dy < radius*radius){//충돌을 했다면
			float dz = sqrtf(radius*radius - dx*dx - dy*dy);//구와 ray가 만나는 곳의 상대 좌표
			*n = dz / sqrtf(radius*radius);
			return dz + z;//ray를 맞은 곳의실제좌표 

		}
		return -INF;
	}
};


__global__ void kernel(unsigned char *ptr,Sphere *s){
	int x = threadIdx.x + blockIdx.x*blockDim.x;
	int y = threadIdx.y + blockIdx.y*blockDim.y;
	int offset = x + y*blockDim.x*gridDim.x;
	float ox = (x - DIM / 2);
	float oy = (y - DIM / 2);
	float r = 0, g = 0, b = 0;
	float maxz = -INF;
	for (int i = 0; i < SPHERES; i++){

		float n;
		float t = s[i].hit(ox, oy, &n);
		float fscale = n;
		if (t>maxz){
			r = s[i].r*fscale;
			g = s[i].g*fscale;
			b = s[i].b*fscale;
		}

	}
	ptr[offset * 4 + 0] = (int)(r * 255);
	ptr[offset * 4 + 1] = (int)(g * 255);
	ptr[offset * 4 + 2] = (int)(b * 255);
	ptr[offset * 4 + 3] = 255;




}


Sphere *s;






extern "C" void raytracing(){

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start,0);
	
	CPUBitmap bitmap(DIM, DIM);
	unsigned char *dev_bitmap;

	cudaMalloc((void**)&dev_bitmap, bitmap.image_size());
	cudaMalloc((void**)&s, sizeof(Sphere)*SPHERES);
	
	Sphere *temp_s = (Sphere*)malloc(sizeof(Sphere)*SPHERES);
	for (int i = 0; i < SPHERES; i++){
		temp_s[i].r = rnd(1.0f);
		temp_s[i].g = rnd(1.0f);
		temp_s[i].b = rnd(1.0f);
		temp_s[i].x = rnd(1000.0f)-500;
		temp_s[i].y = rnd(1000.0f) - 500;
		temp_s[i].z = rnd(1000.0f) - 500;
		temp_s[i].radius = rnd(100.0f)+20;
	}

	cudaMemcpy(s, temp_s, sizeof(Sphere)*SPHERES, cudaMemcpyHostToDevice);
	free(temp_s);
	dim3 grids(DIM / 16, DIM / 16);
	dim3 threads(16, 16);

	kernel << <grids, threads >> >(dev_bitmap,s);

	cudaMemcpy(bitmap.get_ptr(), dev_bitmap, bitmap.image_size(), cudaMemcpyDeviceToHost);
	bitmap.display_and_exit();
	cudaFree(s);
	cudaFree(dev_bitmap);








}