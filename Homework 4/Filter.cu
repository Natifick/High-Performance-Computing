#include <array>
#include <iostream>
#include <stdio.h>
#include <png.h>

#include "utils.cuh"

// ----- Read-write PNG -----

void png_to_arr(png_bytepp& row_pointers, float* array,
                const int& n_rows, const int& n_cols) {
    for (unsigned int y = 0; y < n_rows; y++) {
        png_bytep row = row_pointers[y];
        for (unsigned int x = 0; x < n_cols; x++) {
            png_bytep px = &(row[x * 4]);
            // Grayscale on go
            array[y*n_cols + x] = (px[3] / 255) * (px[0] + px[1] + px[2]) / 3;
            /*printf("%4d, %4d = RGBA(%3d, %3d, %3d, %3d)\n", x, y, px[0], px[1], px[2], px[3]);*/
            //png_byte old[4 * sizeof(png_byte)];
            //memcpy(old, px, sizeof(old));
            
            //px[0] = 255 - old[0];
            //px[1] = 255 - old[1];
            //px[2] = 255 - old[2];
        }
    }
}

void arr_to_png(png_bytepp& row_pointers, float** array,
                const int& n_rows, const int& n_cols) {
    for (unsigned int y = 0; y < n_rows; y++) {
        png_bytep row = row_pointers[y];
        for (unsigned int x = 0; x < n_cols; x++) {
            png_bytep px = &(row[x * 4]);
            px[0] = px[1] = px[2] = (int)(*array)[y*n_cols + x];
            // Grayscale on go
        }
    }
}

void read_png(std::string file_name, int& n_rows, int& n_cols, 
              png_infop& info_ptr, png_bytepp& row_pointers) {
    FILE *fp = fopen(file_name.c_str(), "rb");
    png_structp png_ptr = png_create_read_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    info_ptr = png_create_info_struct(png_ptr);  
    png_init_io(png_ptr, fp);
    png_read_png(png_ptr, info_ptr, PNG_TRANSFORM_IDENTITY, NULL);
    n_cols = png_get_image_width(png_ptr, info_ptr);
    n_rows = png_get_image_height(png_ptr, info_ptr);
    row_pointers = png_get_rows(png_ptr, info_ptr);
    png_destroy_read_struct(&png_ptr, NULL, NULL); 
    fclose(fp);
}

void write_png(std::string file_name, png_infop& info_ptr, png_bytepp& row_pointers) {
    FILE *fp = fopen(file_name.c_str(), "wb");
    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, NULL, NULL, NULL);
    png_init_io(png_ptr, fp);
    png_set_rows(png_ptr, info_ptr, row_pointers);
    png_write_png(png_ptr, info_ptr, PNG_TRANSFORM_IDENTITY, NULL);
    png_destroy_write_struct(&png_ptr, &info_ptr);
    fclose(fp);
}

// ----- GPU code -----
// Check all errors that occur
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
   if (code != cudaSuccess) 
   {
      fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
      if (abort) exit(code);
   }
}

__global__ void convKernel(float* inArray, float* outArray, float* conv,
                           int width, int size, int K) {
    int gid = getGid();
    float result = 0;
    // Sounds not effective, but I can't do much better
    int convpos = 0;
    if (gid < size) {
        int pos;
        for (int i = -K; i <= K; i++) {
            for (int j = -K; j <= K; j++) {
                pos = gid + i*width + j;
                if (pos >= 0 && pos < size) {
                    result += inArray[pos] * conv[convpos];
                }
                convpos++;
            }
        }
        outArray[gid] = result;
    }
}

// Create Gaussian Smoothing convolution

__global__ void makeGaussian(float* kernel, int size, int K, int sigma=1) {
    // We will asssume that function is called from one block
    // The idea is that no one needs convolutions larger than 32x32...
    int gid = getGid();
    __shared__ float sum;
    sum = 0;
    __syncthreads();
    if (gid < size) {
        // I want to make the convolution flat
        int i = (int) (gid / (2*K+1)) - K;
        int j = (gid % (2*K+1)) - K;
        kernel[gid] = expf(-(i*i+j*j)/(2*sigma*sigma));
        atomicAdd(&sum, kernel[gid]);
        // Normalize
        __syncthreads();
        kernel[gid] = kernel[gid] / sum;
    }
}



int main(int argc, char **argv) {
    // Get size of kernel and sigma from args
    int K = 2;
    int sigma = 1;
    if (argc >= 2) {
        K = atoi(argv[1]);
    }
    if (argc >= 3) {
        sigma = atoi(argv[2]);
    }
    int n_rows, n_cols;
    // Some stuff needed by png-reader
    png_infop info_ptr;
    png_bytepp row_pointers;
    std::string in_filename = "image.png", out_filename = "out.png";
// ----- Read data from png -----
    read_png(in_filename, n_rows, n_cols, info_ptr, row_pointers);
    // Now we know the size of the array
    size_t size = n_rows * n_cols;
    float* h_Array = (float*)malloc(sizeof(float) * size);
    png_to_arr(row_pointers, h_Array, n_rows, n_cols);
    
// ----- Process the data -----
    
    // Create convolution kernel
    float* conv;
    int convSize = (2*K+1)*(2*K+1);
    cudaMalloc(&conv, convSize*sizeof(float));
    gpuErrchk(cudaPeekAtLastError());
    makeGaussian<<<1, convSize>>>(conv, convSize, K, sigma);
    cudaDeviceSynchronize();
    gpuErrchk(cudaPeekAtLastError());
    std::cout << size << " " << convSize << std::endl;
    // Create input device-array and move data to it
    float* d_inArray;
    gpuErrchk(cudaMalloc(&d_inArray, sizeof(float) * size));
    gpuErrchk(cudaMemcpy(d_inArray, h_Array, size * sizeof(float), cudaMemcpyHostToDevice));
    
    // Create output device-array to put data to
    float *d_outArray;
    gpuErrchk(cudaMalloc(&d_outArray, sizeof(float) * size));
    cudaDeviceSynchronize();
    // We'll try to optimize the kernel call in here
    convKernel<<<(int)size / 1024, 1024>>>(d_inArray, d_outArray, conv, n_cols, size, K);
    cudaDeviceSynchronize();
    gpuErrchk( cudaPeekAtLastError() );
    cudaMemcpy(h_Array, d_outArray, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaDeviceSynchronize();
// ----- Write data back to another png -----
    arr_to_png(row_pointers, &h_Array, n_rows, n_cols);
    write_png(out_filename, info_ptr, row_pointers);
    
    cudaFree(conv);
    cudaFree(d_inArray);
    cudaFree(d_outArray);
    free(h_Array);
    
    return 0;
}
