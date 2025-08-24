#include<cuda_runtime.h>
#include<iostream>


__global__ void matmul(float* A, float* B, float* C, int M, int N,int K){
  int row = blockDim.y*blockIdx.y+threadIdx.y;
  int col = blockDim.x*blockIdx.x+threadIdx.x;
  int local_row = threadIdx.y;
  int local_col = threadIdx.x;

  if(row<M && col<N){
    float sum=0.0f;
    for(int k=0;k<K;k++){
      sum+=A[row*K+k]*B[k*N+col];
    }
    C[row*N+col] = sum;
  }
}


class DeviceArray{

  public:
  DeviceArray():_ptr(nullptr), row(0), col(0) {}

  ~DeviceArray(){
    release();
  }

  public:
    void reset(int r=0,int c=0){
      release();
      if(r>0 && c>0){
        row = r;
        col = c;
        cudaMalloc(&_ptr,sizeof(float)*r*c);
      }
    }

    float* data(){
      return _ptr;
    }

  private:
  void release(){
    if(_ptr)cudaFree(_ptr);
    _ptr = nullptr;
    row = 0;
    col = 0;
  }

  float* _ptr = nullptr;
  int row;
  int col;
};


class vectorMultiplyPipeline{
  public:
  vectorMultiplyPipeline(){
    cudaStreamCreate(&S[0]);
    cudaStreamCreate(&S[1]);
    cudaStreamCreate(&S[2]);
    cudaStreamCreate(&S[3]);
    dA[0].reset();dA[1].reset();dA[2].reset();dA[3].reset();
    dB[0].reset(); dB[1].reset(); dB[2].reset(); dB[3].reset();
    dC[0].reset();dC[1].reset();dC[2].reset();dC[3].reset();
  }

  ~vectorMultiplyPipeline(){
    dA[0].reset();dA[1].reset();dA[2].reset();dA[3].reset();
    dB[0].reset(); dB[1].reset(); dB[2].reset(); dB[3].reset();
    dC[0].reset();dC[1].reset();dC[2].reset();dC[3].reset();
    for (int i=0;i<4;i++) cudaStreamDestroy(S[i]);
  }


  void toDevice(float* d_arr,float* h_arr, int row0,int rows,int col0,int cols,int ld_src,int ld_dst,cudaStream_t s){
    // 1) Compute where the tile starts on the HOST matrix
    const float* src = h_arr+row0*ld_src+col0;

    // 2)Convert strides-from-rows into BYTES (cudaMemcpy2D works in bytes)
    size_t srcPitch = (size_t) ld_src * sizeof(float);
    size_t dstPitch = (size_t) ld_dst * sizeof(float);

    //3 How many bytes to copy per row of the tile
    size_t widthBytes =(size_t) cols * sizeof(float);

    //4) Issue the 2d copy: copy 'rows' rows, each of "widthBytes" bytes
    // jumping by srcPitch on the host and dstPitch on the device
    cudaMemcpy2DAsync(d_arr,dstPitch,src,srcPitch,widthBytes,rows,cudaMemcpyHostToDevice,s);

  }

    void toHost(float* d_arr,float* s_arr, int row0,int rows,int col0,int cols,int ld_src,int ld_dst,cudaStream_t s){
    // 1) Compute where the tile starts on the HOST matrix
    const float* src = s_arr;

    // 2)Convert strides-from-rows into BYTES (cudaMemcpy2D works in bytes)
    size_t srcPitch = (size_t) ld_src * sizeof(float);
    size_t dstPitch = (size_t) ld_dst * sizeof(float);

    //3 How many bytes to copy per row of the tile
    size_t widthBytes =(size_t) cols * sizeof(float);


    float* dst = d_arr+ld_dst*row0+col0;

    //4) Issue the 2d copy: copy 'rows' rows, each of "widthBytes" bytes
    // jumping by srcPitch on the host and dstPitch on the device
    cudaMemcpy2DAsync(dst,dstPitch,src,srcPitch,widthBytes,rows,cudaMemcpyDeviceToHost,s);

  }


  void operator()(float* h_A, float* h_B, float *h_C, int M, int N, int K){
      // what each stream needs to compute??
      int M0 = M/2;
      int M1 = M-M0;
      int N0 = N/2;
      int N1 = N-N0;

      dA[0].reset(M0,K);
      dA[1].reset(M0,K);
      dA[2].reset(M1,K);
      dA[3].reset(M1,K);

      dB[0].reset(K,N0);
      dB[1].reset(K,N1);
      dB[2].reset(K,N0);
      dB[3].reset(K,N1);

      dC[0].reset(M0,N0);
      dC[1].reset(M0,N1);
      dC[2].reset(M1,N0);
      dC[3].reset(M1,N1);

      //makesure the kernel knows where its writing to C
      int ldA = K, ldB = N;

      //stream0 bounded by M0 N0
      toDevice(dA[0].data(),h_A,0,M0,0,K,ldA,K,S[0]);
      toDevice(dB[0].data(),h_B,0,K,0,N0,ldB,N0,S[0]);

      //stream1 bounded by M0 N1
      toDevice(dA[1].data(),h_A,0,M0,0,K,ldA,K,S[1]);
      toDevice(dB[1].data(),h_B,0,K,N0,N1,ldB,N1,S[1]);


      //stream2 bounded by M1 N0
      toDevice(dA[2].data(),h_A,M0,M1,0,K,ldA,K,S[2]);
      toDevice(dB[2].data(),h_B,0,K,0,N0,ldB,N0,S[2]);


      //stream3 bounded by M1 N1
      toDevice(dA[3].data(),h_A,M0,M1,0,K,ldA,K,S[3]);
      toDevice(dB[3].data(),h_B,0,K,N0,N1,ldB,N1,S[3]);

      dim3 tpb(16,16);
      dim3 bpg0((N0+tpb.x-1)/tpb.x,(M0+tpb.y-1)/tpb.y);
      size_t sharedMem = 0;
      matmul<<<bpg0,tpb,sharedMem,S[0]>>>(dA[0].data(),dB[0].data(),dC[0].data(),M0,N0,K);
      toHost(h_C,dC[0].data(),0,M0,0,N0,N0,N,S[0]);
      
      dim3 bpg1((N1+tpb.x-1)/tpb.x,(M0+tpb.y-1)/tpb.y);
      matmul<<<bpg1,tpb,sharedMem,S[1]>>>(dA[1].data(),dB[1].data(),dC[1].data(),M0,N1,K);
      toHost(h_C,dC[1].data(),0,M0,N0,N1,N1,N,S[1]);


      dim3 bpg2((N0+tpb.x-1)/tpb.x,(M1+tpb.y-1)/tpb.y);
      matmul<<<bpg2,tpb,sharedMem,S[2]>>>(dA[2].data(),dB[2].data(),dC[2].data(),M1,N0,K);
      toHost(h_C,dC[2].data(),M0,M1,0,N0,N0,N,S[2]);


      dim3 bpg3((N1+tpb.x-1)/tpb.x,(M1+tpb.y-1)/tpb.y);
      matmul<<<bpg3,tpb,sharedMem,S[3]>>>(dA[3].data(),dB[3].data(),dC[3].data(),M1,N1,K);
      toHost(h_C,dC[3].data(),M0,M1,N0,N1,N1,N,S[3]);

      
  }
  cudaStream_t stream(int i){
      return S[i];
  }
  private:
  cudaStream_t S[4];
  DeviceArray dA[4],dB[4],dC[4];
};

int main(){
const int N = 500;
const int M = 500;
const int K = 500;
float *hA,*hB,*hC;

cudaMallocHost(&hA,(size_t)M*K*sizeof(float));
cudaMallocHost(&hB,(size_t)K*N*sizeof(float));
cudaMallocHost(&hC,(size_t)M*N*sizeof(float));


for(int i=0;i<(size_t)M*K;i++){
  hA[i]=1.0f;//why f?

}

for(int i=0;i<(size_t)K*N;i++){
  hB[i]=2.0f;//why f?
}

vectorMultiplyPipeline multiply;
multiply(hA,hB,hC,M,N,K);
for (int i = 0; i < 4; ++i) cudaStreamSynchronize(multiply.stream(i));
std::cout << "C[0,0]=" << hC[0] << "  C[last,last]="
          << hC[(M-1)*N + (N-1)] << "\n";


// after multiply(...)
cudaFreeHost(hA);
cudaFreeHost(hB);
cudaFreeHost(hC);
}