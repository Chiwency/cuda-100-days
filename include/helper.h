#include <iostream>
#include <cuda_runtime.h>

#define CHECK_CUDA(func)                                                       \
{                                                                              \
    cudaError_t status = (func);                                               \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define CHECK_LAST_ERROR()                                                     \
{                                                                              \
    cudaError_t status = cudaGetLastError();                                   \
    if (status != cudaSuccess) {                                               \
        printf("CUDA API failed at line %d with error: %s (%d)\n",             \
               __LINE__, cudaGetErrorString(status), status);                  \
        return EXIT_FAILURE;                                                   \
    }                                                                          \
}

#define BLOCK_SIZE 1024
#define WARP_SIZE 32


void printMatrix(int *matrix, int rows, int cols, const char *name) {
    printf("Matrix %s:\n", name);
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            printf("%d ", matrix[i * cols + j]);
        }
        printf("\n");
    }
}

void printVector(int *vector, int N, const char *name) {
    printf("Vector %s:\n", name);
    for (int i = 0; i < N; i++) {
        printf("%d ", vector[i]);
    }
    printf("\n");
}
