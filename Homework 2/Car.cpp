#include <fstream>
#include <iostream>
#include <stdlib.h>
#include <string>
#include <cassert>
#include <omp.h>

#define RGB_COMPONENT_COLOR 255

int N_THREADS = 12;

struct PPMPixel {
  int red;
  int green;
  int blue;
};

typedef struct {
  int x, y, all;
  PPMPixel *data;
} PPMImage;

void readPPM(const char *filename, PPMImage &img){
    std::ifstream file(filename);
    if (file) {
        std::string s;
        int rgb_comp_color;
        file >> s;
        if (s != "P3") {
            std::cout << "error in format" << std::endl;
            exit(9);
        }
        file >> img.x >> img.y;
        file >> rgb_comp_color;
        img.all = img.x * img.y;
        std::cout << s << std::endl;
        std::cout << "x=" << img.x << " y=" << img.y << " all=" << img.all
                  << std::endl;
        img.data = new PPMPixel[img.all];
        for (int i = 0; i < img.all; i++) {
            file >> img.data[i].red >> img.data[i].green >> img.data[i].blue;
        }
    } else {
        std::cout << "the file:" << filename << "was not found" << std::endl;
    }
    file.close();
}

void moveCar(PPMImage &img) {
    PPMPixel *new_arr = new PPMPixel[img.all + img.y];
    #pragma omp parallel for num_threads(N_THREADS) collapse(2)
    for (int i = 0; i < img.y; i++) {
        for (int j = 1; j <= img.x; j++) {
            new_arr[(img.x+1) * i + j] = img.data[img.x * i + j-1];
        }
    }
    #pragma omp parallel for num_threads(N_THREADS) collapse(2)
    for (int i = 0; i < img.y; i++) {
        for (int j = 1; j < img.x; j++) {
            img.data[img.x * i + j] = new_arr[(img.x+1) * i + j];
        }
    }
    #pragma omp parallel for num_threads(N_THREADS)
    for (int i = 0; i < img.y; i++) {
        img.data[i*img.x] = new_arr[i*(img.x+1)+img.x];
    }
}

bool operator==(PPMPixel first, PPMPixel second) {
    return (first.red == second.red) &&
           (first.green == second.green) &&
           (first.blue == second.blue);
}

void writePPM(const char *filename, PPMImage &img) {
    std::ofstream file(filename, std::ofstream::out);
    file << "P3" << std::endl;
    file << img.x << " " << img.y << " " << std::endl;
    file << RGB_COMPONENT_COLOR << std::endl;

    for (int i = 0; i < img.all; i++) {
        file << img.data[i].red << " " << img.data[i].green << " "
             << img.data[i].blue << (((i + 1) % img.x == 0) ? "\n" : " ");
    }
    file.close();
}

int main(int argc, char *argv[]) {
    PPMImage image;
    readPPM("car.ppm", image);
    // Let's remember which was the first car
    PPMPixel *initialCar = new PPMPixel[image.all];
    for (int i=0;i<image.all;i++) {initialCar[i]=image.data[i];}
    // Move car around
    double start, end;
    start = omp_get_wtime();
    for (int i = 0; i < image.x; i++) {
        moveCar(image);
    }
    end = omp_get_wtime();
    std::cout << "Total time: " << end - start << std::endl;
    // The car should be in the initial position
    for (int i=0;i<image.all;i++) {assert(initialCar[i]==image.data[i]);}
    writePPM("new_car.ppm", image);
    return 0;
}




