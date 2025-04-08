#include "helper.h"

// __syncthreads()  can't synchronize threads in different blocks, so it can't 
// keep correct in multiple blocks compute.
__global__ void partialSum(int *A, int *B, int N)
{
    int threadIndex = blockDim.x * blockIdx.x + threadIdx.x;
    int i = threadIndex * 2;
    for (int stride = 1; stride < gridDim.x * blockDim.x; stride *= 2)
    {
        if (threadIndex % stride == 0)
        {
            A[i] += A[i + stride];
        }
        __syncthreads();
    }
    if (i == 0)
    {
        *B = A[0];
    }
}

int main()
{
    int N = 10261;
    int *A, *B;
    cudaMallocManaged(&A, N * sizeof(int));
    cudaMallocManaged(&B, sizeof(int));

    for (int i = 0; i < N; i++)
    {
        A[i] = i;
    }

    int blockSize = min(((N + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE, BLOCK_SIZE);
    int gridSize = (N + blockSize - 1) / blockSize;
    std::cout << "blockSize: " << blockSize << std::endl;
    std::cout << "gridSize: " << gridSize << std::endl;
    std::cout << "accutal result: " << N * (N - 1) / 2 << std::endl;
    partialSum<<<gridSize, blockSize>>>(A, B, N);
    CHECK_LAST_ERROR();
    cudaDeviceSynchronize();

    std::cout << "B: " << *B << std::endl;
}