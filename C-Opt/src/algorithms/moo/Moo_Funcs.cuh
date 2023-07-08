//
// Created by nasser on 06/07/23.
//

#ifndef C_OPT_MOO_FUNCS_CUH
#define C_OPT_MOO_FUNCS_CUH

#include <iostream>
#include <cuda_runtime.h>

// CUDA kernel function
__global__ void evaluateFunction(float* values, int n) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;

    if (index < n) {
        // Evaluate your function here
        float result = values[index] * values[index]; // Squaring the input

        // Store the result back to the input array
        values[index] = result;
    }
}

class CudaFunctionEvaluator {
public:
    void evaluate(float* values, int n) {
        float* devValues;
        cudaMalloc((void**)&devValues, n * sizeof(float));
        cudaMemcpy(devValues, values, n * sizeof(float), cudaMemcpyHostToDevice);

        int blockSize = 256;
        int numBlocks = (n + blockSize - 1) / blockSize;

        evaluateFunction<<<numBlocks, blockSize>>>(devValues, n);

        cudaMemcpy(values, devValues, n * sizeof(float), cudaMemcpyDeviceToHost);
        cudaFree(devValues);
    }
};

int main() {
    int n = 10;
    float values[n] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0};

    CudaFunctionEvaluator evaluator;
    evaluator.evaluate(values, n);

    std::cout << "Results: ";
    for (int i = 0; i < n; i++) {
        std::cout << values[i] << " ";
    }
    std::cout << std::endl;

    return 0;
}


#endif //C_OPT_MOO_FUNCS_CUH
