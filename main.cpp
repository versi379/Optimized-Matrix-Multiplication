#include "ErrorHandling.h"
#include "MatrixUtilities.h"
#include "DisplayGpuInfo.h"

#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cublasXt.h>

#include <iostream>
#include <string.h>
#include <vector>
#include <chrono>

using std::cout;
using std::endl;

//Verify the GPU/Device and Host Result
#define VERIFY_ARRAYS 1

//32 or 16 bits floats arrays for the GPU arrays
#define A_32_BIT 1
#define B_32_BIT 1
#define C_32_BIT 1 //Always 32 bit


int main()
{
    DisplayGpuInfo();

    constexpr int rowsA = 10000;
    constexpr int rank = 100;
    constexpr int colsB = 10000;

    int colsA = rank;
    int rowsB = rank;

    cout << "Arrays: rows X colums" << endl;
    cout << "A = " << rowsA << " X " << colsA << endl;
    cout << "B = " << rowsB << " X " << colsB << endl;
    cout << "C = " << rowsA << " X " << colsB << endl
        << endl;

    int m = rowsA;
    int k = rank;
    int n = colsB;
    int lda = rowsA;
    int ldb = rowsB;
    int ldc = rowsA;

    if (m % 4 == 0)
    { //rowsA
        cout << "m % 4 == 0, best performance" << endl;
    }

    if (k % 8 == 0)
    { // rank = colsA = rowsB
        cout << "k % 8 == 0, best performance" << endl;
    }

    if (lda % 8 == 0)
    { // rowsA ,16F A,B
        cout << "lda % 8 == 0, best performance" << endl; 
    }

    if (ldb % 8 == 0)
    { // rank = colsA = rowsB, 16F A,B
        cout << "ldb % 8 == 0, best performance" << endl;
    }

    if (ldc % 4 == 0)
    { // rank = colsA = rowsB, 16F A,B
        cout << "ldc % 4 == 0, best performance" << endl;
        
    }

    size_t sizeA = rowsA * rank;
    size_t sizeB = rank * colsB;
    size_t sizeC = rowsA * colsB;

#if A_32_BIT
    float* A = new float[sizeA];
#else //16 Bit
    half* A;
    A = (half*)malloc(sizeA * sizeof(half));
#endif

#if B_32_BIT
    float* B = new float[sizeB];
#else //16 Bit
    half* B;
    B = (half*)malloc(sizeB * sizeof(half));
#endif

#if C_32_BIT
    float* C = new float[sizeC];
#else
    #error "Result array must always be 32 bit."
#endif


#if VERIFY_ARRAYS
    float* verifyC = new float[sizeC];
#endif

    cout << "A is " << MatrixSizeMB(A, rowsA, rank) << "MB" << endl;
    cout << "B is " << MatrixSizeMB(B, rank, colsB) << "MB" << endl;
    cout << "C is " << MatrixSizeMB(C, rowsA, colsB) << "MB" << endl;

    //Fill arrays on host
    FillMatrix(A, m, k);
    FillMatrix(B, k, n);

    cublasHandle_t handle;
    CUBLASS_HANDLE_ERROR(cublasCreate(&handle));
    CUBLASS_HANDLE_ERROR(cublasSetMathMode(handle, cublasMath_t::CUBLAS_DEFAULT_MATH));

    cublasMath_t mathMode;
    CUBLASS_HANDLE_ERROR(cublasGetMathMode(handle, &mathMode));

    if (mathMode == CUBLAS_DEFAULT_MATH) {
        cout << "OK math mode: CUBLAS_DEFAULT_MATH" << endl;
    }

    cudaDataType_t d_A_mode;
    cudaDataType_t d_B_mode;
    cudaDataType_t d_C_mode;

#if A_32_BIT
    float* d_A;
    d_A_mode = CUDA_R_32F;
#else
    half* d_A;
    d_A_mode = CUDA_R_16F;
#endif

#if B_32_BIT
    float* d_B;
    d_B_mode = CUDA_R_32F;
#else
    half* d_B;
    d_B_mode = CUDA_R_16F;
#endif

#if C_32_BIT
    float* d_C;
    d_C_mode = CUDA_R_32F;
#else
    half* d_C;
    d_C_mode = CUDA_R_16F;
#endif

    HANDLE_ERROR(cudaMalloc(&d_A, sizeA * sizeof(A[0])));
    HANDLE_ERROR(cudaMalloc(&d_B, sizeB * sizeof(B[0])));
    HANDLE_ERROR(cudaMalloc(&d_C, sizeC * sizeof(C[0])));

    //Transfering arrays from host to device
    HANDLE_ERROR(cudaMemcpy(d_A, A, sizeA * sizeof(A[0]), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_B, B, sizeB * sizeof(B[0]), cudaMemcpyHostToDevice));
    HANDLE_ERROR(cudaMemcpy(d_C, C, sizeC * sizeof(C[0]), cudaMemcpyHostToDevice));

    float alpha = 1.0f;
    float beta = 0.0f;

    if (intptr_t(d_A) % 16 == 0)
    {
        cout << "\n\nintptr_t(d_A) % 16 == 0, best performance\n";
    }

    if (intptr_t(d_B) % 16 == 0)
    {
        cout << "intptr_t(d_B) % 16 == 0, best performance\n";
    }

    if (intptr_t(d_C) % 16 == 0)
    {
        cout << "intptr_t(d_C) % 16 == 0, best performance\n";
    }

    std::cout << "\nStarting GPU matrix multiplication\n";
    auto start = std::chrono::high_resolution_clock::now();
    CUBLASS_HANDLE_ERROR(cublasGemmEx(handle, CUBLAS_OP_N, CUBLAS_OP_N, m, n, k, &alpha,
        d_A, d_A_mode, lda,
        d_B, d_B_mode, ldb,
        &beta, d_C, d_C_mode, lda, CUDA_R_32F, CUBLAS_GEMM_ALGO0_TENSOR_OP));

    HANDLE_ERROR(cudaDeviceSynchronize());
    auto stop = std::chrono::high_resolution_clock::now();
    
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

    cout << "GPU Time: " << duration.count() << " ms" << std::endl;

    HANDLE_ERROR(cudaMemcpy(C, d_C, sizeC * sizeof(C[0]), cudaMemcpyDeviceToHost));
    HANDLE_ERROR(cudaDeviceSynchronize());

    HANDLE_ERROR(cudaFree(d_A));
    HANDLE_ERROR(cudaFree(d_B));
    HANDLE_ERROR(cudaFree(d_C));
    CUBLASS_HANDLE_ERROR(cublasDestroy(handle));

    float* f_C = new float[sizeC];

    cout << "\nStarting CPU matrix multiplication" << endl;
    start = std::chrono::high_resolution_clock::now();
    
    MatrixMulCPU(f_C, A, B, rowsA, rank, colsB); //Full float matrix multiply CPU
    
    stop = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::milliseconds>(stop - start);

    cout << "CPU Time: " << duration.count() << " ms" << std::endl;

#if VERIFY_ARRAYS
    // std::cout << "\n";
    // std::cout << "\n";
    // PrintMatrix(A, rowsA,rank);

    // std::cout << "\n";
    // std::cout << "\n";
    // PrintMatrix(B, rank,colsB);

    // std::cout << "\n";
    // std::cout << "HOST \n";
    // PrintMatrix(f_C, rowsA, colsB);


    // std::cout << "\n";
    // std::cout << "GPU Result\n";
    // PrintMatrix(C, rowsA, colsB);

    VerifyArrays(C, f_C, rowsA, colsB);
#endif

    delete[] A;
    delete[] B;
    delete[] C;

#if VERIFY_ARRAYS
    free(verifyC);
#endif

    delete[] f_C;

    return 0;
}
