# MMD Project

## Overview

This is a project designed to quantify discrepancy bewteen two datasets using the [Maximum Mean Discrepancy](https://en.wikipedia.org/wiki/Kernel_embedding_of_distributions#Measuring_distance_between_distributions) (*e.g.* data/MC discrepancies).

## Idea

The idea is the following:

- Let's consider two datasets of size $(n,m) \in \mathbb{N}^*$, $X = (x_1, x_2, ..., x_n)$ and $Y = (y_1, y_2, ..., y_m)$.

- Each element of the dataset represents an event. Those events are themselves vectors of size $d \in \mathbb{N}^*$, $x_i = (x_{i1}, x_{i2}, ..., x_{id})$ and $y_j = (y_{j1}, y_{j2}, ..., y_{jd})$, where $d$ is the number of features.

- One can then consider that each event is a point in a $d$-dimensional probability space, identically and idependently distributed (i.i.d.) according to a probability distribution $P_X$ for the $(x_i)$ and $P_Y$ for the $(y_i)$.

- Following this, one can then ask the question: how different are the two distributions $P_X$ and $P_Y$?

- For that, one can compute the Maximum Mean Discrepancy (MMD) between the two datasets. The MMD is defined as:

$$
\text{MMD}(X, Y) = \frac{1}{n^2} \sum_{i=1}^n \sum_{j=1}^n k_{\lambda}(x_i, x_j) + \frac{1}{m^2} \sum_{i=1}^m \sum_{j=1}^m k_{\lambda}(y_i, y_j) - \frac{2}{nm} \sum_{i=1}^n \sum_{j=1}^m k_{\lambda}(x_i, y_j) 
$$

where $k$ is a positive definite *kernel function*. Let's consider the Gaussian kernel:

$$
k_{\lambda}(x_i, y_j) = \frac{1}{\pi^{d/2}} \sum_{l=1}^d \exp\left(-\frac{|x_{il} - y_{jl}|^2}{\lambda_l^2}\right)
$$

and $\lambda \in \mathbb{R}^d$ is called the bandwidth of the kernel function.

- In this case, $\lambda$ is chosen using the Silverman's rule of thumb:

$$
\lambda_l = \frac{1}{2} \cdot \text{IQR}(X_l) \cdot n^{-1/5}
$$

where $\text{IQR}(X_l)$ is the interquartile range of the $l$-th feature of the dataset $X$.