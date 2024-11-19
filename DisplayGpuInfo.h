#ifndef DISPLAY_GPU_INFO_H
#define DISPLAY_GPU_INFO_H

#include "ErrorHandling.h"

#include <iostream>

using std::endl;

static void DisplayGpuInfo() {
    size_t freeMem;
    size_t totalMem;

    const int kb = 1024;
    const int mb = kb * kb;

    std::cout << "CUDA version: " << CUDART_VERSION << endl;

    int devCount;
    HANDLE_ERROR(cudaGetDeviceCount(&devCount));
    std::cout << "CUDA Devices: " << endl << endl;

    if(devCount == 0) {
        std::cerr << "No Nvidia capable GPU found exiting\n";
        exit(EXIT_FAILURE);
    }

    for (int i = 0; i < devCount; ++i)
    {
        cudaDeviceProp props;
        HANDLE_ERROR(cudaGetDeviceProperties(&props, i));
        HANDLE_ERROR(cudaMemGetInfo(&freeMem, &totalMem));

        size_t freeMB = freeMem >> 20;
        size_t totalMB = totalMem >> 20;

        std::cout << i << ": " << props.name << ": " << props.major << "." << props.minor << endl;
        std::cout << "  Global memory:   " << totalMB << "mb" << endl;
        std::cout << "  Available memory: " << freeMB << "mb" << endl;
        std::cout << "  Shared memory:   " << props.sharedMemPerBlock / kb << "kb" << endl;
        std::cout << "  Constant memory: " << props.totalConstMem / kb << "kb" << endl;
        std::cout << "  Block registers: " << props.regsPerBlock << endl << endl;

        std::cout << "  Warp size:         " << props.warpSize << endl;
        std::cout << "  Threads per block: " << props.maxThreadsPerBlock << endl;
        std::cout << "  Max block dimensions: [ " << props.maxThreadsDim[0] << ", " << props.maxThreadsDim[1] << ", " << props.maxThreadsDim[2] << " ]" << endl;
        std::cout << "  Max grid dimensions:  [ " << props.maxGridSize[0] << ", " << props.maxGridSize[1] << ", " << props.maxGridSize[2] << " ]" << endl;
        std::cout << endl;
    }
}

#endif
