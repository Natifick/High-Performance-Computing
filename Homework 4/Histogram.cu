#include <array>
#include <iostream>
#include <stdio.h>
#include <png.h>

#include "utils.cuh"

// ----- Read-write PNG -----

void png_to_arr(png_bytepp& row_pointers, int** array,
                const int& n_rows, const int& n_cols) {
    *array = (int*)malloc(sizeof(int) * n_cols * n_rows);
    for (unsigned int y = 0; y < n_rows; y++) {
        png_bytep row = row_pointers[y];
        for (unsigned int x = 0; x < n_cols; x++) {
            png_bytep px = &(row[x * 4]);
            // Grayscale on go
            (*array)[y*n_cols + x] = (int) (px[3] / 255) * (px[0] + px[1] + px[2]) / 3;
            /*printf("%4d, %4d = RGBA(%3d, %3d, %3d, %3d)\n", x, y, px[0], px[1], px[2], px[3]);*/
            //png_byte old[4 * sizeof(png_byte)];
            //memcpy(old, px, sizeof(old));
            
            //px[0] = 255 - old[0];
            //px[1] = 255 - old[1];
            //px[2] = 255 - old[2];
        }
    }
}

void arr_to_png(png_bytepp& row_pointers, int** array,
                const int& n_rows, const int& n_cols) {
    for (unsigned int y = 0; y < n_rows; y++) {
        png_bytep row = row_pointers[y];
        for (unsigned int x = 0; x < n_cols; x++) {
            png_bytep px = &(row[x * 4]);
            px[0] = px[1] = px[2] = (*array)[y*n_cols + x];
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

__global__ void zeroKernel(float* array, int size) {
    int gid = getGid();
    if (gid < size) {
        array[gid] = 0;
    }
}

__global__ void stencilKernel(float* inArray, float* outArray, int size) {
    int gid = getGid();
    if (gid < size) {
        outArray[gid] = inArray[gid] * inArray[gid];
    }
}

__global__ void adderKernel(float* array, int size) {
    int gid = getGid();
    if (gid < size) {
        //array[0] = array[0] + 1;
        atomicAdd(&array[0], 1);
    }
}

// CPU code
int main() {
    int n_rows, n_cols;
    // Some stuff needed by png-reader
    png_infop info_ptr;
    png_bytepp row_pointers;
    std::string in_filename = "image.png", out_filename = "out.png";
    // ----- Read data from png -----
    read_png(in_filename, n_rows, n_cols, info_ptr, row_pointers);
    int* array;
    png_to_arr(row_pointers, &array, n_rows, n_cols);
    
    // ----- Process the data -----
    
    // TODO

    // ----- Write data back to another png -----
    arr_to_png(row_pointers, &array, n_rows, n_cols);
    write_png(out_filename, info_ptr, row_pointers);
    
    return 0;
}
