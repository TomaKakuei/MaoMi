// Evaluate the same MATLAB-v4 Green function and dense coefficients in FP64.
// Each query accumulates all channels, sharing distances and logarithms.
extern "C" __global__ void green_apply(
    const double* p, const double* c, const double* q, double* out,
    int n, int m) {
  int t = blockIdx.x * blockDim.x + threadIdx.x;
  if (t >= m) return;
  double x = q[2*t], y = q[2*t+1];
  double acc[NVALUES] = {};
  for (int i = 0; i < n; ++i) {
    double dx = x-p[2*i], dy = y-p[2*i+1];
    double r2 = dx*dx + dy*dy;
    double g = r2 > 0. ? r2*(0.5*log(r2)-1.) : 0.;
    #pragma unroll
    for (int j = 0; j < NVALUES; ++j) acc[j] += g*c[NVALUES*i+j];
  }
  #pragma unroll
  for (int j = 0; j < NVALUES; ++j) out[NVALUES*t+j] = acc[j];
}
