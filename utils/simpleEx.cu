/*
* Add a constant to a vector.
*/
__global__ void addToVector(float * pi, float c, int vecLen)  {
   int idx = blockIdx.x * blockDim.x + threadIdx.x;
   if (idx < vecLen) {
       pi[idx] += c;
   }
}