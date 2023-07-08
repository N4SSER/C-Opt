//
// Created by nasser on 06/07/23.
//

#include "Moo_Funcs.cuh"

MOO_func::MOO_func(KernelFunction kernelFunc) : kernel_function(kernelFunc)
{

}

MOO_func::~MOO_func()
{
    cudaFree(d_X);
    cudaFree(d_Y);
}

void MOO_func::evaluate(int **X, int *Y, int numElements)
{
    cudaMalloc((void**)&d_X, sizeof(int*) * numElements);
    cudaMemcpy(d_X, X, sizeof(int*) * numElements, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_Y, sizeof(int) * numElements);

    int blockSize = 256;
    int numBlocks = (numElements + blockSize - 1) / blockSize;

    kernel_function<<<numBlocks, blockSize>>>(d_X, d_Y, numElements);

    cudaMemcpy(Y, d_Y, sizeof(int) * numElements, cudaMemcpyDeviceToHost);
}
