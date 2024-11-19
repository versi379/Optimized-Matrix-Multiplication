#ifndef MATRIX_UTILITIES_H
#define MATRIX_UTILITIES_H

#include <cuda_fp16.h>

#include <random>
#include <iostream>
#include <string>
#include <fstream>

using std::cout;
using std::endl;
using std::ifstream;

template <typename T>
size_t MatrixSizeMB(const T* A, const size_t rows, const size_t cols)
{
    size_t sizeA = rows * cols * sizeof(A[0]);

    return sizeA >> 20;
}

template <typename T>
void PrintMatrix(const T* A, const size_t rows, const size_t cols)
{
    for (size_t i = 0; i < rows; ++i)
    {
        for (size_t j = 0; j < cols; ++j)
        {
            cout << A[j * rows + i] << " ";
        }
        cout << endl;
    }
    cout << endl;
}

static std::default_random_engine generator;
//Keep in mind that if you have 16 bits floats keep the max number within range.
static std::uniform_real_distribution<float> distribution(1.00f, 99.00f);

template <typename T>
void FillMatrix(T* A, const size_t rows, const size_t cols)
{
    for (size_t i = 0; i < rows; ++i)
    {
        for (size_t j = 0; j < cols; ++j)
        {
            T r = distribution(generator);
            A[j * rows + i] = r;
        }
    }
}

template <typename T, typename T1>
void MatrixMulCPU(T* C, const T1* A, const T1* B, const size_t rowsA, const size_t colsA, const size_t colsB)
{
    for (size_t i = 0; i < rowsA; ++i) {
        for (size_t j = 0; j < colsB; ++j)
        {
            T sum = 0.0;

            for (size_t k = 0; k < colsA; ++k)
            {
                //Important must be double
                double a = A[k * rowsA + i];
                double b = B[j * colsA + k];

                sum += (a * b);
            }

            C[j * rowsA + i] = sum;
        }
    }
}

template <typename T>
void VerifyArrays(const T* A, const T* B, const size_t rows, const size_t cols)
{
    bool flag = true;
    for (size_t i = 0; i < rows; ++i)
    {
        for (size_t j = 0; j < cols; ++j)
        {
            const T value1 = A[j * rows + i];
            const T value2 = B[j * rows + i];
            double error = 0.01;
            if (value1 - value2 > error)
            {
                flag = false;
                cout << "Error too big " << value1 - value2 << endl;
            }
        }
    }

    if (flag)
    {
        cout << "Host and Device Matrices MATCH" << endl;
    }
    else
    {
        cout << "Host and Device Matrices DONT MATCH " << endl;
    }
}

#endif
