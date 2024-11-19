# Optimized-Matrix-Multiplication

This implementation leverages the NVIDIA CUDA framework and the cuBLAS library to optimize matrix multiplication using the cublasGemmEx function. By utilizing the power of GPU acceleration and advanced features like Tensor Cores (if supported), the computation is significantly faster compared to traditional CPU-based methods.

# Overview

This application performs basic matrix multiplication: \( A \times B = C \).

- **Matrix dimensions:**
  - \( A \): `rowsA x rank`  
  - \( B \): `rank x colsB`  
  - \( C \): `rowsA x colsB`

- **Array representation:**  
  Matrices are represented as 2D arrays using single raw pointers, e.g.:

  ```c++
  float* A = new float[sizeA];
  ```

- **Accessing elements:**  
  Elements are accessed using the following pattern:

  ```c++
  for (size_t i = 0; i < rows; ++i)
  {
      for (size_t j = 0; j < cols; ++j)
      {
          cout << A[j * rows + i] << " ";
      }
      cout << endl;
  }
  ```

- **Data types:**  
  Matrices \( A \) and \( B \) can use either 16-bit or 32-bit floats, but the result matrix \( C \) is always 32-bit.

- **Performance:**  
  GPU (device) execution significantly outperforms CPU (host, single-threaded) execution.  
  Starting with cuBLAS version 11, Tensor Cores are utilized automatically. More details are available in the [NVIDIA cuBLAS documentation](https://docs.nvidia.com/cuda/cublas/#tensor-core-usage).

# Build Instructions

## Linux

1. **Install CUDA toolkit dependencies:**

   ```bash
   sudo apt install nvidia-cuda-toolkit
   ```

2. **Build the application:**

   ```bash
   make all
   ```

3. **Run the executable:**

   ```bash
   ./main.out
   ```