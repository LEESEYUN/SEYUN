#include <iostream>
#include "cuda.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include "../../common/gl_helper.h"


#define DIM 1024
#define PI 3.1415926535897932f

extern "C" void julia();

int main(){
	julia();
	return 0;
}


