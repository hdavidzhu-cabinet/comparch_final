# OpenCL Experiments

This repo goes through 2 OpenCL tutorials to lean how OpenCL programs are built and executed. The following tutorials are here:

- Filtering: https://anteru.net/blog/2012/11/03/2009/
- Matrix multiplication: http://gpgpu-computing4.blogspot.com/2009/09/matrix-multiplication-2-opencl.html

## Instructions

Here are instructions to execute our programs.

```bash
# First, run cmake to create our Makefile.
cmake CMakeLists.txt

# For image filtering, run
make && ./gpu_comparch_final

# For matrix multiplication, run
make && ./matrix_mul
```
