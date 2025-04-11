#include "helper.h"

// layerNorm(A) = (A - mean(A)) / stddev(A), used a lot in transformers
// It ensures that each row of A has zero mean and stddev=1 
__global__ void layerNorm(float *A,float *B,int rows,int cols) {
    int t=blockDim.x*blockIdx.x+threadIdx.x;
    float sum=0;
    float sum_sq=0;
    if (t>=rows) {
        return;
    }
    for (int i=0;i<cols;i++) {
        float temp=A[t*rows+i];
        sum+=temp;
        sum_sq+=temp*temp;
    }

    float mean=sum/cols;
    float stddev=sqrtf(sum_sq/cols-mean*mean + 1e-7);
    for (int i=0;i<cols;i++) {
        B[t*rows+i]=(A[t*rows+i]-mean)/stddev;
    }
}


int main() {
    int rows=5,cols=5;
    float *A,*B;

    cudaMallocManaged(&A,rows*cols*sizeof(float));
    cudaMallocManaged(&B,rows*cols*sizeof(float));

    for (int i = 0 ; i < rows*cols; i++) {
        A[i] = rand() / (float)RAND_MAX;
    }

    int blockSize=min(rows,BLOCK_SIZE);
    int gridSize=(rows+blockSize-1)/blockSize;
    layerNorm<<<gridSize,blockSize>>>(A,B,rows,cols);

    cudaDeviceSynchronize();

    printMatrix(A,rows,cols,"A");
    printMatrix(B,rows,cols,"B");
    
}