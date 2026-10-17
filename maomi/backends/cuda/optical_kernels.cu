// CUDA C kernels. NVRTC compilation via CuPy; FFTs remain NVIDIA cuFFT.
// No fast-math: preserve the selected reference arithmetic within float32 error.
extern "C" __global__ void gather_screens(
    const float2* src, const float* edge, const int* offsets, float2* dst,
    int B, int Z, int H, int W, int N) {
    long long t=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    long long total=(long long)B*Z*N*N;
    if(t>=total) return;
    int y=t%N, x=(t/N)%N, z=(t/(N*N))%Z, b=t/((long long)N*N*Z);
    float2 a=src[((long long)z*H+offsets[2*b]+x)*W+offsets[2*b+1]+y];
    float e=edge[x*N+y]; dst[t]=make_float2(a.x*e,a.y*e);
}

extern "C" __global__ void resample_power(
    const float2* field, const int* ix, const float* wx,
    const int* iy, const float* wy, const float* axial, float* out,
    int B,int Z,int H,int W,int X,int Y,int KX,int KY,int order,float scale) {
    long long t=(long long)blockIdx.x*blockDim.x+threadIdx.x;
    if(t>=(long long)B*X*Y*Z) return;
    int z=t%Z,y=(t/Z)%Y,x=(t/(Z*Y))%X,b=t/((long long)Z*Y*X);
    float value=0;
    for(int i=0;i<KX;++i){
        int sx=ix[x*KX+i]; float a=wx[x*KX+i];
        if(a==0) continue;
        float row=0;
        for(int j=0;j<KY;++j){
            float w=wy[y*KY+j]; if(w==0) continue;
            int sy=iy[y*KY+j];
            float2 u=field[((long long)b*Z+z)*H*W+sx*W+sy];
            float power=u.x*u.x+u.y*u.y;
            if(order==1) row+=w*power;
            else if(order==2) row+=w*power*power;
            else row+=w*power*power*power;
        }
        value+=a*row;
    }
    out[t]=scale*value*axial[z];
}
