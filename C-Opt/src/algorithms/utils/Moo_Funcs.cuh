//
// Created by nasser on 06/07/23.
//

#ifndef C_OPT_MOO_FUNCS_CUH
#define C_OPT_MOO_FUNCS_CUH


typedef void (*KernelFunction)(int**, int*, int);

class FunctionEvaluator {
public:
    FunctionEvaluator(KernelFunction kernelFunc);
    ~FunctionEvaluator();

    void evaluate(int** X, int* Y, int numElements);

private:
    int** d_X;
    int* d_Y;
    KernelFunction kernel_function;
};



#endif //C_OPT_MOO_FUNCS_CUH
