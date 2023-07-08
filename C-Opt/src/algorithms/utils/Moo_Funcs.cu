//
// Created by nasser on 06/07/23.
//

#include "Moo_Funcs.cuh"

FunctionEvaluator::FunctionEvaluator(KernelFunction kernelFunc) : kernel_function(kernelFunc)
{

}

FunctionEvaluator::~FunctionEvaluator()
{
    cudaFree(d_X);
    cudaFree(d_Y);
}

void FunctionEvaluator::evaluate(int **X, int *Y, int numElements)
{
    cudaMalloc((void**)&d_X, sizeof(int*) * numElements);
    cudaMemcpy(d_X, X, sizeof(int*) * numElements, cudaMemcpyHostToDevice);

    cudaMalloc((void**)&d_Y, sizeof(int) * numElements);

    int blockSize = 256;
    int numBlocks = (numElements + blockSize - 1) / blockSize;

    kernel_function<<<numBlocks, blockSize>>>(d_X, d_Y, numElements);

    cudaMemcpy(Y, d_Y, sizeof(int) * numElements, cudaMemcpyDeviceToHost);
}
