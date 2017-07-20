#include <iostream>
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"






extern "C" void raytracing();




int main(){
	raytracing();




	return 0;
}