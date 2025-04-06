#include <stdio.h>

int main()
{
    int device = 0;

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    printf("  Device name: %s\n", prop.name);
    // printf("  Memory Clock Rate (KHz): %d\n", prop.memoryClockRate);
    // printf("  Memory Bus Width (bits): %d\n", prop.memoryBusWidth);
    // printf("  Peak Memory Bandwidth (GB/s): %f\n",
    //         2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
    // printf("  Total global memory: %f GB\n", (float)prop.totalGlobalMem / (1024 * 1024 * 1024));
    printf("  Shared memory per block: %.1f KB\n", (float)prop.sharedMemPerBlock / 1024);
    // printf("  Registers per block: %d\n", prop.regsPerBlock);
    //printf("  Warp size: %d\n", prop.warpSize);
    printf("  Max threads per block: %d\n", prop.maxThreadsPerBlock);
    printf("  Max threads per multiprocessor: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("  Number of multiprocessors: %d\n", prop.multiProcessorCount);
    // printf("  Compute capability: %d.%d\n", prop.major, prop.minor);

    // Calculate cores per SM based on architecture
    int cores_per_sm = 0;
    switch (prop.major) {
        case 2:  // Fermi
            cores_per_sm = 32; break;
        case 3:  // Kepler
            cores_per_sm = 192; break;
        case 5:  // Maxwell
            cores_per_sm = 128; break;
        case 6:  // Pascal
            if (prop.minor == 1) cores_per_sm = 128;  // GP100
            else cores_per_sm = 64;  // GP104, GP106, etc.
            break;
        case 7:  // Volta, Turing
            cores_per_sm = 64; break;
        case 8:  // Ampere
            cores_per_sm = 64; break;
        case 9:  // Hopper
            cores_per_sm = 128; break;
        default:
            printf("Unknown architecture\n");
            return 1;
    }
    printf("  Cores per SM: %d\n", cores_per_sm);
    return 0;
}
