#include "helper.h"

__global__ void matrixAdd(int *A, int *B, int *C, int N)
{
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    int col = blockIdx.y * blockDim.y + threadIdx.y;
    int idx = row * N + col;
    if (row >= N || col >= N)
    {
        return;
    }

    C[idx] = A[idx] + B[idx];
}

int main()
{
    int N = 10;
    int *A, *B, *C;
    CHECK_CUDA(cudaMallocManaged(&A, N * N * sizeof(int)));
    CHECK_CUDA(cudaMallocManaged(&B, N * N * sizeof(int)));
    CHECK_CUDA(cudaMallocManaged(&C, N * N * sizeof(int)));

    for (int i = 0; i < N * N; i++)
    {
        A[i] = i;
        B[i] = i;
        C[i] = 0;
    }

    dim3 dimBlock(WARP_SIZE, BLOCK_SIZE / WARP_SIZE);
    dim3 dimGrid(ceil((N * N + dimBlock.x - 1) / dimBlock.x), 1);
    matrixAdd<<<dimGrid, dimBlock>>>(A, B, C, N);
    cudaDeviceSynchronize();
    CHECK_LAST_ERROR();
    // Print matrix A
    printf("Matrix A:\n");
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            printf("%d ", A[i * N + j]);
        }
        printf("\n");
    }

    // Print matrix B
    printf("Matrix B:\n");
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            printf("%d ", B[i * N + j]);
        }
        printf("\n");
    }

    // Print matrix C
    printf("Matrix C:\n");
    for (int i = 0; i < N; i++)
    {
        for (int j = 0; j < N; j++)
        {
            printf("%d ", C[i * N + j]);
        }
        printf("\n");
    }

    return 0;
}
