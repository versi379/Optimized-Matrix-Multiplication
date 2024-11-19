#ifndef ERRORHANDLING_H
#define ERRORHANDLING_H

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <iostream>

static void HandleError(cudaError_t err,
	const char *file,
	int line) {
	if (err != cudaSuccess) {
		std::cerr << cudaGetErrorString(err) << " in " << file << " at line " << line <<std::endl;
		exit(EXIT_FAILURE);
	}
}

#define HANDLE_ERROR( err ) (HandleError( err, __FILE__, __LINE__ ))


static void cuBLASSHandleError(cublasStatus_t error,
	const char *file,
	int line) {
	if (error != CUBLAS_STATUS_SUCCESS) {
		switch (error)
		{
		case CUBLAS_STATUS_NOT_INITIALIZED:
			fprintf(stderr, "CUBLAS_STATUS_NOT_INITIALIZED in %s at line %d\n",
				file, line);
			exit(EXIT_FAILURE);

		case CUBLAS_STATUS_ALLOC_FAILED:
			fprintf(stderr, "CUBLAS_STATUS_ALLOC_FAILED in %s at line %d\n",
				file, line);
			exit(EXIT_FAILURE);

		case CUBLAS_STATUS_INVALID_VALUE:
			fprintf(stderr, "CUBLAS_STATUS_INVALID_VALUE in %s at line %d\n",
				file, line);
			exit(EXIT_FAILURE);

		case CUBLAS_STATUS_ARCH_MISMATCH:
			fprintf(stderr, "CUBLAS_STATUS_ARCH_MISMATCH in %s at line %d\n",
				file, line);
			exit(EXIT_FAILURE);

		case CUBLAS_STATUS_MAPPING_ERROR:
			fprintf(stderr, "CUBLAS_STATUS_MAPPING_ERROR in %s at line %d\n",
				file, line);
			exit(EXIT_FAILURE);

		case CUBLAS_STATUS_EXECUTION_FAILED:
			fprintf(stderr, "CUBLAS_STATUS_EXECUTION_FAILED in %s at line %d\n",
				file, line);
			exit(EXIT_FAILURE);

		case CUBLAS_STATUS_INTERNAL_ERROR:
			fprintf(stderr, "CUBLAS_STATUS_INTERNAL_ERROR in %s at line %d\n",
				file, line);
			exit(EXIT_FAILURE);
	
		case CUBLAS_STATUS_NOT_SUPPORTED:
			fprintf(stderr, "CUBLAS_STATUS_NOT_SUPPORTED in %s at line %d\n",
				file, line);
			exit(EXIT_FAILURE);
	
		case CUBLAS_STATUS_LICENSE_ERROR:
			fprintf(stderr, "CUBLAS_STATUS_LICENSE_ERROR in %s at line %d\n",
				file, line);
			exit(EXIT_FAILURE);
		}
	}
}

#define CUBLASS_HANDLE_ERROR( err ) (cuBLASSHandleError( err, __FILE__, __LINE__ ))

#endif
