#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <sstream>
#include <time.h>
#include <sys/time.h>

#ifdef __APPLE__
	#include "OpenCL/opencl.h"
#else
	#include "CL/cl.h"
#endif

// Allocates a matrix with random float entries.
void randomInit(float* data, int size) {
   for (int i = 0; i < size; ++i)
   data[i] = rand() / (float) RAND_MAX;
}

void checkError(cl_int error) {
	if (error != CL_SUCCESS) {
		std::cerr << "OpenCL call failed with error " << error << std::endl;
		std::exit(1);
	}
}

std::string LoadKernel(const char* name) {
	std::ifstream in (name);
	std::string result (
		(std::istreambuf_iterator<char> (in)),
		std::istreambuf_iterator<char> ());
	return result;
}

cl_program CreateProgram(const std::string& source, cl_context context) {
	// http://www.khronos.org/registry/cl/sdk/1.1/docs/man/xhtml/clCreateProgramWithSource.html
	size_t lengths [1] = { source.size () };
	const char* sources [1] = { source.data () };

	cl_int error = 0;
	cl_program program = clCreateProgramWithSource (context, 1, sources, lengths, &error);
	checkError(error);

	return program;
}

// Following: http://gpgpu-computing4.blogspot.com/2009/09/matrix-multiplication-2-opencl.html
int main() {

  int MY_LOCAL_WORK_SIZE = 1;
  int MY_MATRIX_SIZE = 2048;

  // Allocate memory.
  unsigned int width_A = MY_MATRIX_SIZE;
  unsigned int height_A = MY_MATRIX_SIZE;
  unsigned int size_A = width_A * height_A;
  unsigned int mem_size_A = sizeof(float) * size_A;
  float* matrix_A = (float*) malloc(mem_size_A);

  unsigned int width_B = MY_MATRIX_SIZE;
  unsigned int height_B = MY_MATRIX_SIZE;
  unsigned int size_B = width_B * height_B;
  unsigned int mem_size_B = sizeof(float) * size_B;
  float* matrix_B = (float*) malloc(mem_size_B);

  // Initialize the host memory.
  randomInit(matrix_A, size_A);
  randomInit(matrix_B, size_B);

  // Print out the arrays.
  // printf("\n\nMatrix A\n");
  // for (int i = 0; i < size_A; i++) {
  //   printf("%f", matrix_A[i]);
  //   if (((i + 1) % width_A) == 0) {
  //     printf("\n");
  //   }
  // }

  // printf("\n\nMatrix B\n");
  // for (int i = 0; i < size_B; i++) {
  //   printf("%f", matrix_B[i]);
  //   if (((i + 1) % width_B) == 0) {
  //     printf("\n");
  //   }
  // }

  // Allocate host memory for C.
  unsigned int size_C = width_A * height_B;
  unsigned int mem_size_C = sizeof(float) * size_C;
  float* host_C = (float*) malloc(mem_size_C);

  // Allocate device (GPU) memory.
  float* device_A;
  float* device_B;

  size_t dataBytes;
  size_t kernelLength;
  cl_int errorCode;

  // Initialize OpenCL.
  cl_context clGPUContext = clCreateContextFromType(0, CL_DEVICE_TYPE_GPU, NULL, NULL, &errorCode);
  checkError(errorCode);

  // Get a list of GPU devices associated with context.
  errorCode = clGetContextInfo(clGPUContext, CL_CONTEXT_DEVICES, 0, NULL, &dataBytes);
  cl_device_id *clDevices = (cl_device_id *) malloc(dataBytes);
  errorCode |= clGetContextInfo(clGPUContext, CL_CONTEXT_DEVICES, dataBytes, clDevices, NULL);
  checkError(errorCode);

  // Create a command queue.
  cl_command_queue clCommandQueue = clCreateCommandQueue(clGPUContext, clDevices[0], 0, &errorCode);
  checkError(errorCode);

  cl_mem d_C = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE, mem_size_C, NULL, &errorCode);
  cl_mem d_A = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mem_size_A, matrix_A, &errorCode);
  cl_mem d_B = clCreateBuffer(clGPUContext, CL_MEM_READ_WRITE | CL_MEM_COPY_HOST_PTR, mem_size_B, matrix_B, &errorCode);

  // Load and build the OpenCL kernel.
	cl_program clMatrixMul = CreateProgram(LoadKernel("kernels/matrix_multiplication.cl"), clGPUContext);
	checkError(clBuildProgram(clMatrixMul, 0, NULL, NULL, NULL, NULL));

	// http://www.khronos.org/registry/cl/sdk/1.1/docs/man/xhtml/clCreateKernel.html
	cl_kernel kernel = clCreateKernel(clMatrixMul, "matrixMul", &errorCode);
	checkError(errorCode);

  // Launch OpenCL kernel.
  size_t localWorkSize[2], globalWorkSize[2];

  int wA = width_A;
  int wC = width_A; // TODO: Check this.

  errorCode = clSetKernelArg(kernel, 0, sizeof(cl_mem), (void *)&d_C);
  errorCode |= clSetKernelArg(kernel, 1, sizeof(cl_mem), (void *)&d_A);
  errorCode |= clSetKernelArg(kernel, 2, sizeof(cl_mem), (void *)&d_B);
  errorCode |= clSetKernelArg(kernel, 3, sizeof(int), (void *)&wA);
  errorCode |= clSetKernelArg(kernel, 4, sizeof(int), (void *)&wC);
  checkError(errorCode);

  localWorkSize[0] = MY_LOCAL_WORK_SIZE;
  localWorkSize[1] = MY_LOCAL_WORK_SIZE;
  globalWorkSize[0] = MY_MATRIX_SIZE;
  globalWorkSize[1] = MY_MATRIX_SIZE;

  typedef unsigned long long u64;
  u64 u64useconds;
  struct timeval tv;
  gettimeofday(&tv, NULL);
  u64useconds = (1000000*tv.tv_sec) + tv.tv_usec;
  fprintf(stdout, "%lu\n", (unsigned long) u64useconds);

  // Enqueue the program. It will run, and we check for errors.
  errorCode = clEnqueueNDRangeKernel(clCommandQueue, kernel, 2, NULL, globalWorkSize, localWorkSize, 0, NULL, NULL);
  checkError(errorCode);

  // Retrieve result from device.
  errorCode = clEnqueueReadBuffer(clCommandQueue, d_C, CL_TRUE, 0, mem_size_C, host_C, 0, NULL, NULL);
  checkError(errorCode);

  gettimeofday(&tv, NULL);
  u64useconds = (1000000*tv.tv_sec) + tv.tv_usec;
  fprintf(stdout, "%lu\n", (unsigned long) u64useconds);

  // 9. print out the results
  // printf("\n\nMatrix C (Results)\n");
  // for(int i = 0; i < size_C; i++) {
  //   printf("%f ", host_C[i]);
  //   if(((i + 1) % width_A) == 0) {
  //    printf("\n");
  //   }
  // }
  // printf("\n");

  // 10. clean up memory
  free(matrix_A);
  free(matrix_B);
  free(host_C);

  clReleaseMemObject(d_A);
  clReleaseMemObject(d_C);
  clReleaseMemObject(d_B);

  free(clDevices);
  free(clMatrixMul);
  clReleaseContext(clGPUContext);
  clReleaseKernel(kernel);
  // clReleaseProgram(clMatrixMul);
  clReleaseCommandQueue(clCommandQueue);
}
