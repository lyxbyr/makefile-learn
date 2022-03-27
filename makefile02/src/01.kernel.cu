/*
 * @Author: liaoyixiong 
 * @Date: 2021-12-25 17:48:34 
 * @Last Modified by:   liaoyixiong 
 * @Last Modified time: 2021-12-25 17:48:34 
 */

#include "stdio.h"
#include <cuda_runtime.h>



/*
cuda核函数
__global__ 核函数的前缀定义
   - 使用__global__修饰的函数，必须是void无返回值
   - __global__核函数修饰，必须是nvcc编译才有效，否则无效
   - __global__修饰的函数， 使用name<<<grid, block, memory, stream>>>(params)启动核函数 
         - 启动在host, 但执行在device
定义如下：
__device__, 函数执行在设备上
__global__, 函数执行在设备上，但是调用在host上，定义核函数的符号
__host__, 函数执行在host上，调用也在host上

__device__ 修饰的函数， 只能在设备上执行，设备上调用(例如核函数内调用)
    - nvidia提供了很多内置设备函数，比如日常的cos, sin之类的
        - 在nvidia团队中， 不同的内置函数的api接口版本号，被称为计算能力

*/




// sigmoid 不能够使用 sigmoid<<<1, 3>>>这种启动它
// 也不能直接 sigmoid(0.1)
// 只能在核函数内调用他

__device__ float sigmoid(float value) {
  return 1 / (1 + exp(-value));
}





__global__ void compute(float* a, float* b, float* c) {

  /*  
    线程layout的概念， 启动文档线程会被设计为gird和block, 如同提高的参数一样
    这个layohut的概念是虚拟的，通过cuda驱动实现真实硬件映射，抽象了一个中间值（调度层）
      - 如果我们有4353 Core
      - 如果我们需要启动5000个线程
      - 抽象层它会把5000个线程安排到各个Core中执行， 根据情况来执行次数
      - 每次调度单位为WarpSize, 如果启动的线程不足，也会执行WarpSize, 不过Core是非激活状态而已
    
    需要启动多少个线程，通过girdDim和blockDim告诉它
    线程数= gridDim.x * gridDim.y * gridDim.z * blockDim.x * blockDim.y * blockDim.z 


    2147483647  是有符号整数(int)的最大值
    65535       是无符号整数(int16m short)的最大值
    gridDim的最大值范围 ： (x,y,z): (2147483647, 65535, 65535)
    blockDim的最大值范围： (x,y,z): (1024, 1024, 64)



    它的定义在device_launch_parameters.h
    uint3 __device_builtin__ __STORAGE__ threadIdx;
    uint3 __device_builtin__ __STORAGE__ blockIdx;
    dim3 __device_builtin__ __STORAGE__ blockDim;
    dim3 __device_builtin__ __STORAGE__ gridDim;
    int __device_builtin__ __STORAGE__ warpSize;

    获取线程ID, 进行数据操作
    数据索引， 是通过blockIdx和threadIdx计算得到
    girdDim告诉你Grid的大小， blockDim告诉你block大小
    blockIdx告诉你所在Grid内的索引， threadIdx告诉你所在block内的索引
    把gridDim和blockDim设想为一个tensor
    则：
    gridDim的shape  = gridDim.z * gridDim.y * gridDim.x
    blockDim的shape = blockDim.z * blockDim.y * blockDim.x
    最终的启动线程的shape维度为  gridDim.z * gridDim.y * gridDim.x * blockDim.z * blockDim.y * blockDim.x

    如果启动的线程是6个维度的tensor, 那么索引，也可以类似
    blockIdx.z, blockIdx.y, blockIdx.x, threadIdx.z, threadIdx.y, threadIdx.x

    仅仅是在这个场景需要把6个维度索引变为连续的内存单元索引
    如果有6个维度a, b, c, d, e, f和6个位置的索引u, v, w, x, y, z
    a     u
    b     v  
    c     w
    d     x  
    e     y
    f     z  
    position = ((((u * b + v) * c + w) * d + x) * e + y) * f + z

  */

  // gridDim  = 1 * 1 * 1
  // blockDim = 3 * 1 * 1 

  int position = blockDim.x * blockIdx.x + threadIdx.x;
  c[position] = a[position] * sigmoid(b[position]);
}


int main() {

  const int num = 3;
  float a[num] = {1, 2, 3};
  float b[num] = {5, 7, 9};
  float c[num] = {0};
  
  size_t size_array = sizeof(c);
  float* device_a = nullptr;
  float* device_b = nullptr;
  float* device_c = nullptr;
  
  cudaMalloc(&device_a, size_array);
  cudaMalloc(&device_b, size_array);
  cudaMalloc(&device_c, size_array);

  cudaMemcpy(device_a, a, size_array, cudaMemcpyHostToDevice);
  cudaMemcpy(device_b, b, size_array, cudaMemcpyHostToDevice);

  compute<<<dim3(1), dim3(3)>>>(device_a, device_b, device_c);

  cudaMemcpy(c, device_c, size_array, cudaMemcpyDeviceToHost);

  for (int i = 0; i < num; ++i) {
    printf("c[%d] = %f\n", i, c[i]);
  }
  return 0;






}