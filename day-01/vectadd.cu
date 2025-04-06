
#include "helper.h"

#define CHECK_CUDA_ERROR(val) check_cuda_error((val), #val, __FILE__, __LINE__)

__global__ void vectorAdd(int *a, int *b, int *c, int n)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        c[i] = a[i] + b[i];
    }
}

using namespace std;

int main()
{
    const int N = 10;
    int *a, *b, *c;

    cudaMallocManaged(&a, N * sizeof(int));
    cudaMallocManaged(&b, N * sizeof(int));
    cudaMallocManaged(&c, N * sizeof(int));
    for (int i = 0; i < N; i++)
    {
        a[i] = i;
        b[i] = i;
    }

    int blockSize = 1024;
    int gridSize = (N + blockSize - 1) / blockSize;
    vectorAdd<<<gridSize, blockSize>>>(a, b, c, N);
    cudaDeviceSynchronize();
    CHECK_LAST_ERROR();
    for (int i = 0; i < N; i++)
    {
        cout << c[i] << " ";
    }

    cout << endl;
    return 0;
}