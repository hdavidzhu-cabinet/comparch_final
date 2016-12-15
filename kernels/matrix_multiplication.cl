/* matrix_multiplication.cl
 * Matrix multiplication: C = A * B.
 * Device code.
 */

// OpenCL Kernel
__kernel void matrixMul(
  __global float* C,
  __global float* A,
  __global float* B,
  int width_A, int width_B) {

   // 2D Thread ID
   int thread_X = get_global_id(0);
   int thread_Y = get_global_id(1);

   // value stores the element that is
   // computed by the thread
   float value = 0;
   for (int k = 0; k < width_A; ++k) {
      float elementA = A[thread_Y * width_A + k];
      float elementB = B[k * width_B + thread_X];
      value += elementA * elementB;
   }

   // Write the matrix to device memory each
   // thread writes one element
   C[thread_Y * width_A + thread_X] = value;
}
