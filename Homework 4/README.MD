# CUDA Homework 4

## 1. Laplace equation  

1. solve the Laplace equation using CUDA. (10 points)
2. draw the heatmap of the solution u(x,y) using anything you want (10 points)

Here are the equation and the boundary conditions:

$$\frac{\delta^2u}{\delta^2x}+\frac{\delta^2u}{\delta^2y}=0$$

Boundary conditions:
* $u(x=0, y) = u(x=1, y) = 0$
* $u(x, y=0) = 1$
* $u(x, y=1) = 0$

A good explanation of how to discretize the Laplace equation is given hereLinks to an external site. (starting from page 3). In principle you can use any of the 3 approaches:

* discretize the equation and get (NM)^2 sparse linear system and inverse the matrix;
* discretize the equation and get (NM)^2 sparse linear system and solve it iteratively;
* try to achieve the steady-state solution of the corresponding heat equation with some initial conditions (replace 0 with du/dt and kind of do what we did in seminar 4 until the solution more or less stops changing in time).

## 2. Filtering

Take an arbitrary image and apply two types of filters to it using CUDA.

* Blurring filter (have at least 2 blurring filters - they can differ in stencil size or the values of the filter matrix for example) (15 points)
* Median filter (try to achieve the cartoonish effect) (15 points)

## 3. Histogram

You need to plot a  histogram for a chosen picture. For this purpose:

1. take any picture -> grayscale it -> calculate the histogram for the picture using CUDA (10 points)
2. plot the resulting histogram using anything you want (10 points)
