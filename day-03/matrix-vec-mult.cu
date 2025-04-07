#include "helper.h"

// matrix N*N *  vertor N*1
__global__ void matrixVecMult(int *A, int *B, int *C, int N) {
    int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= N) {
        return;
    }
    int sum=0;
    for (int i=0; i<N; i++){
        sum += A[row*N+i]*B[i];
    }

    C[row] = sum;
}

int main(){
    int N = 10;
    int *A, *B, *C;
    cudaMallocManaged(&A, N * N * sizeof(int));
    cudaMallocManaged(&B, N * sizeof(int));
    cudaMallocManaged(&C, N * sizeof(int));

    for (int i = 0; i < N; i++) {
        for (int j =0; j < N; j++){
            A[i * N + j] = 1;
        }
        B[i] = i;
    }

    int padN=((N+WARP_SIZE-1)/WARP_SIZE)*WARP_SIZE;
    matrixVecMult<<<1,padN>>>(A, B, C, N);

    cudaDeviceSynchronize();
    printMatrix(A, N, N, "A");
    printVector(B, N, "B");
    printVector(C, N, "C");

    
}