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

// shared memory is dedicate to each block
__global__ void partialSumWithSharedMemory(int *A, int *B, int N)
{
    // extern in cpp declares a variable without defining it, means it will be defined elsewhere, in cuda, it's used for dynamic shared memory allocation.
    extern __shared__ int temp[];
    int segment = 2 * blockDim.x * blockIdx.x;
    int t = threadIdx.x;
    int i = segment + t;
    temp[t] = A[i] + A[i + blockDim.x];
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
    {
        __syncthreads();
        if (t < stride)
        {
            temp[t] += temp[t + stride];
        }
    }
    if (t == 0)
    {
        atomicAdd(B, temp[0]);
    }
}

int main()
{
    int N = 20470;
    int *A, *B;
    cudaMallocManaged(&A, N * sizeof(int));
    cudaMallocManaged(&B, sizeof(int));

    for (int i = 0; i < N; i++)
    {
        A[i] = i;
    }

    int blockSize = min(((N / 2 + WARP_SIZE - 1) / WARP_SIZE) * WARP_SIZE, BLOCK_SIZE);
    int gridSize = (N / 2 + blockSize - 1) / blockSize;
    std::cout << "blockSize: " << blockSize << std::endl;
    std::cout << "gridSize: " << gridSize << std::endl;
    std::cout << "accutal result: " << N * (N - 1) / 2 << std::endl;
    // partialSum<<<gridSize, blockSize>>>(A, B, N);
    partialSumWithSharedMemory<<<gridSize, blockSize, blockSize * sizeof(int)>>>(A, B, N);
    CHECK_LAST_ERROR();
    cudaDeviceSynchronize();

    std::cout << "B: " << *B << std::endl;
}