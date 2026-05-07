#include <stdlib.h>
#include <math.h>
#include "vector.h"
#include "config.h"
#include <cuda_runtime.h>

#define EPSILON 1e-9
// Kernel for computing gravitational forces
__global__ void compute_gravitational_forces(vector3* pos, double* mass, vector3* accels, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;

	if (i >= N || j >= N) return;

	

	if (i ==j) {
		FILL_VECTOR(accels[i*N + j], 0,0,0);
		return;
	}

	vector3 distance;
    double magnitude_sq = 0.0;

    for (int k = 0; k < 3; k++) {
        distance[k] = pos[i][k] - pos[j][k];
        magnitude_sq += distance[k] * distance[k];
    }

    if (magnitude_sq < EPSILON) return;

    double magnitude = sqrt(magnitude_sq);

    double accelmag = -GRAV_CONSTANT * mass[j] / magnitude_sq;

    FILL_VECTOR(
        accels[i * N + j],
        accelmag * distance[0] / magnitude,
        accelmag * distance[1] / magnitude,
        accelmag * distance[2] / magnitude
    );
}
// Kernel for reducing the gravitational forces
__global__ void reduce_accels(vector3* accels, vector3* accel_sum, int N) {

	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= N) return;

	vector3 sum = {0, 0, 0};
	for (int j = 0; j < N; j++) {
		for (int k = 0; k < 3; k++) {
			sum[k] += accels[i * N + j][k];
		}
	}
 accel_sum[i] = sum;
}

//compute: Updates the positions and locations of the objects in the system based on gravity.
//Parameters: None
//Returns: None
//Side Effect: Modifies the hPos and hVel arrays with the new positions and accelerations after 1 INTERVAL
void compute(){
	 int N = NUMENTITIES;

    vector3 *d_pos, *d_accels, *d_accel_sum;
    double *d_mass;

    cudaMalloc(&d_pos, sizeof(vector3) * N);
    cudaMalloc(&d_accels, sizeof(vector3) * N * N);
    cudaMalloc(&d_mass, sizeof(double) * N);
    cudaMalloc(&d_accel_sum, sizeof(vector3) * N);

    cudaMemcpy(d_pos, hPos, sizeof(vector3) * N, cudaMemcpyHostToDevice);
    cudaMemcpy(d_mass, mass, sizeof(double) * N, cudaMemcpyHostToDevice);

    dim3 threads(16, 16);
    dim3 blocks((N + 15) / 16, (N + 15) / 16);

    compute_gravitational_forces<<<blocks, threads>>>(d_pos, d_mass, d_accels, N);
    cudaDeviceSynchronize();

    reduce_accels<<<(N + 255) / 256, 256>>>(d_accels, d_accel_sum, N);
    cudaDeviceSynchronize();

    vector3* accel_sum = (vector3*)malloc(sizeof(vector3) * N);

    cudaMemcpy(accel_sum, d_accel_sum, sizeof(vector3) * N, cudaMemcpyDeviceToHost);

    for (int i = 0; i < N; i++) {
        for (int k = 0; k < 3; k++) {
            hVel[i][k] += accel_sum[i][k] * INTERVAL;
            hPos[i][k] += hVel[i][k] * INTERVAL;
        }
    }

    cudaFree(d_pos);
    cudaFree(d_accels);
    cudaFree(d_mass);
    cudaFree(d_accel_sum);
    free(accel_sum);
}
