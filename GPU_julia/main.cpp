
#include <iostream>
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../../common/gl_helper.h"

extern "C" void cudaFunction();

int main(){
	cudaFunction();


	return 0;
}
