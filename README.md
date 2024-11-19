# Cuda-Matrix-Multiplication

A basic example of how to do matrix multiplcation with cuda and cuBLAS library.
In order for the example to work you must have an Nvidia GPU supporting CUDA.  
Multiplcations are done by using ```cublasGemmEx``` method.

# Overview

This simple app does the basic matrix multplication A X B = C

A = rowsA * rank  
B = rank * colsB  
C = rowsA * colsB  

Arrays are 2d and are declared using single raw pointers

```c++
float* A = new float[sizeA];
```

In order to access them we follow this trick

```C++
for (size_t i = 0; i < rows; ++i)
{
    for (size_t j = 0; j < cols; ++j)
    {
        cout << A[j * rows + i] << " ";
    }
    cout << endl;
}
```

The could be either 16 or 32 bit floats.
The result array is always 32 bit.

The time difference beetween the GPU(device) and CPU(host and single threaded) are very big.

Also Nvidia states the after cuBLAS version 11 tensor cores will be used automatically, [link](https://docs.nvidia.com/cuda/cublas/#tensor-core-usage).

# Build

## Linux

Make sure you have all the required dependencies for cuda to compile and work.

```bash
sudo apt install nvidia-cuda-toolkit
```

Build
```bash
make all
```

Run the executable

```bash
./main.out
```
