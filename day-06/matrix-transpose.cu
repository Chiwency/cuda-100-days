#include "helper.h"

__global__ void transpose(int *A, int *B, int row, int col){
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i<row && j<col){
        B[j*row+i] = A[i*col+j];
    }
}

int main(){
    int row=102499;
    int col=10241;
    int *A, *B;
    cudaMallocManaged(&A, row*col*sizeof(int));
    cudaMallocManaged(&B, row*col*sizeof(int));

    for (int i=0;i<row*col;i++){
        A[i] = i;
    }
    
    dim3 dimBlock(32,32);
    dim3 dimGrid(ceil(row/32.0),ceil(col/32.0));
    transpose<<<dimGrid,dimBlock>>>(A,B,row,col);
    cudaDeviceSynchronize();
    for (int i=0;i<row;i++){
        for (int j=0;j<col;j++){
            if (A[i*col+j]!=B[j*row+i]){
                std::cout<<"Error"<<std::endl;
                return 0;
            }
        }
    }
    std::cout<<"Success"<<std::endl;
    return 0;
}