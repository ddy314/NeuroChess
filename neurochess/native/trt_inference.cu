#include "trt_inference.hpp"
#include <NvInferRuntime.h>
#include <NvInferPlugin.h>
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

namespace {
void checked(cudaError_t error) {
  if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
struct Logger : nvinfer1::ILogger {
  void log(Severity severity, const char *message) noexcept override {
    if (severity <= Severity::kWARNING) std::cerr << "TensorRT: " << message << '\n';
  }
};
struct Stream {
  cudaStream_t value=nullptr;
  Stream() { checked(cudaStreamCreateWithFlags(&value,cudaStreamNonBlocking)); }
  ~Stream() { if(value) {cudaStreamSynchronize(value);cudaStreamDestroy(value);} }
  operator cudaStream_t() const {return value;}
};
template<class T, bool host = false> struct Buffer {
  T *ptr = nullptr;
  explicit Buffer(size_t count) {
    if constexpr(host) checked(cudaMallocHost(reinterpret_cast<void **>(&ptr), count * sizeof(T)));
    else checked(cudaMalloc(reinterpret_cast<void **>(&ptr), count * sizeof(T)));
  }
  ~Buffer() { if constexpr(host) cudaFreeHost(ptr); else cudaFree(ptr); }
  Buffer(const Buffer &) = delete;
};
__device__ float sum256(float value, float *work) {
  int lane = threadIdx.x;
  work[lane] = value; __syncthreads();
  for(int width = 128; width; width >>= 1) {
    if(lane < width) work[lane] += work[lane + width];
    __syncthreads();
  }
  float result = work[0]; __syncthreads(); return result;
}
__device__ int best256(float value, bool eligible, float *work, int *indices) {
  int lane = threadIdx.x;
  work[lane] = eligible ? value : -INFINITY;
  indices[lane] = eligible ? lane : 256; __syncthreads();
  for(int width = 128; width; width >>= 1) {
    if(lane < width && (work[lane + width] > work[lane] ||
       (work[lane + width] == work[lane] && indices[lane + width] < indices[lane]))) {
      work[lane] = work[lane + width]; indices[lane] = indices[lane + width];
    }
    __syncthreads();
  }
  int result = indices[0]; __syncthreads(); return result;
}
// Stable legal top-five and the existing Q/WDL fusion formula, no search change.
__global__ void fuse_heads(const float *policy, const float *qvalues,
    const float *wdl, const int32_t *ids, float *output, float *raw) {
  int row = blockIdx.x, lane = threadIdx.x;
  __shared__ float work[256], maximum, value, confidence, w[3];
  __shared__ int indices[256], old;
  bool legal = ids[row * 256 + lane] >= 0, selected = false;
  float p = legal ? policy[row * 256 + lane] : -1e9f;
  float q = qvalues[row * 256 + lane];
  for(int k = 0; k < 5; ++k) {
    int best = best256(p, legal && !selected, work, indices);
    if(k == 0 && lane == 0) { maximum = work[0]; old = best; }
    selected = selected || lane == best; __syncthreads();
  }
  bool supported = selected && legal && p >= maximum - 3.f;
  float exponential = legal ? __expf(p - maximum) : 0.f;
  float probability = exponential / sum256(exponential, work);
  float weight = supported ? probability : 0.f;
  weight /= sum256(weight, work);
  float mean = sum256(weight * q, work);
  if(lane == 0) {
    float maximum_w = fmaxf(wdl[row * 3], fmaxf(wdl[row * 3 + 1], wdl[row * 3 + 2]));
    float total = 0.f;
    for(int k = 0; k < 3; ++k) { w[k] = __expf(wdl[row * 3 + k] - maximum_w); total += w[k]; }
    float entropy = 0.f;
    for(int k = 0; k < 3; ++k) { w[k] /= total; entropy -= w[k] * __logf(fmaxf(w[k], 1e-8f)); }
    value = w[0] - w[2];
    confidence = fminf(1.f, fmaxf(0.f, 1.f - entropy / 1.0986122886681098f));
  }
  __syncthreads();
  float fused = p + 2.f * (.25f + .75f * confidence) * (q - mean) * float(supported);
  output[row * 515 + lane] = fused;
  output[row * 515 + 256 + lane] = supported ? .5f * q + .5f * value : NAN;
  raw[row * 515 + lane] = p;
  raw[row * 515 + 256 + lane] = q;
  int best = best256(fused, legal, work, indices);
  float support_count = sum256(float(supported), work);
  if(lane == 0) {
    output[row * 515 + 512] = value;
    output[row * 515 + 513] = support_count;
    output[row * 515 + 514] = best != old;
    for(int k = 0; k < 3; ++k) raw[row * 515 + 512 + k] = w[k];
  }
}
struct Slot {
  int batch;
  std::unique_ptr<nvinfer1::IExecutionContext> context;
  Buffer<int32_t, true> ht, hi;
  Buffer<float, true> hm, ho, hr;
  Buffer<int32_t> dt, di;
  Buffer<float> dm, policy, q, wdl, fused, raw;
  cudaGraph_t graph = nullptr;
  cudaGraphExec_t executable = nullptr;
  Slot(nvinfer1::ICudaEngine &engine, int n)
      : batch(n), context(engine.createExecutionContext()), ht(n*64), hi(n*256),
        hm(n*12), ho(n*515), hr(n*515), dt(n*64), di(n*256), dm(n*12),
        policy(n*256), q(n*256), wdl(n*3), fused(n*515), raw(n*515) {
    if(!context) throw std::runtime_error("Cannot create TensorRT context");
    std::fill_n(ht.ptr,n*64,0); std::fill_n(hi.ptr,n*256,0); std::fill_n(hm.ptr,n*12,0.f);
    for(auto item : {std::pair<const char *,int>{"tokens",64}, {"meta",12}, {"moves",256}}) {
      nvinfer1::Dims shape; shape.nbDims=2; shape.d[0]=n; shape.d[1]=item.second;
      if(!context->setInputShape(item.first,shape)) throw std::runtime_error("Unsupported input shape");
    }
    for(auto item : {std::pair<const char *,void *>{"tokens",dt.ptr}, {"meta",dm.ptr},
         {"moves",di.ptr}, {"policy",policy.ptr}, {"q",q.ptr}, {"wdl_logits",wdl.ptr}})
      if(!context->setTensorAddress(item.first,item.second)) throw std::runtime_error("Cannot bind engine tensor");
    for(auto item : {std::pair<const char *,int>{"policy",256}, {"q",256}, {"wdl_logits",3}}) {
      auto shape=context->getTensorShape(item.first);
      if(shape.nbDims!=2 || shape.d[0]!=n || shape.d[1]!=item.second ||
         engine.getTensorDataType(item.first)!=nvinfer1::DataType::kFLOAT)
        throw std::runtime_error("Unexpected output shape/type");
    }
  }
  ~Slot() { if(executable) cudaGraphExecDestroy(executable); if(graph) cudaGraphDestroy(graph); }
  void enqueue(cudaStream_t stream) {
    checked(cudaMemcpyAsync(dt.ptr,ht.ptr,batch*64*sizeof(int32_t),cudaMemcpyHostToDevice,stream));
    checked(cudaMemcpyAsync(di.ptr,hi.ptr,batch*256*sizeof(int32_t),cudaMemcpyHostToDevice,stream));
    checked(cudaMemcpyAsync(dm.ptr,hm.ptr,batch*12*sizeof(float),cudaMemcpyHostToDevice,stream));
    if(!context->enqueueV3(stream)) throw std::runtime_error("TensorRT enqueue failed");
    fuse_heads<<<batch,256,0,stream>>>(policy.ptr,q.ptr,wdl.ptr,di.ptr,fused.ptr,raw.ptr);
    checked(cudaGetLastError());
    checked(cudaMemcpyAsync(ho.ptr,fused.ptr,batch*515*sizeof(float),cudaMemcpyDeviceToHost,stream));
  }
};
} // namespace

namespace nc_native {
struct TrtInference::Impl {
  Logger logger;
  Stream stream;
  std::unique_ptr<nvinfer1::IRuntime> runtime;
  std::unique_ptr<nvinfer1::ICudaEngine> engine;
  std::vector<std::unique_ptr<Slot>> slots;
  bool graphs;
  int last = 0;
  Impl(const std::string &path, int maximum, bool use_graphs) : graphs(use_graphs) {
    if(maximum<1 || maximum>16) throw std::runtime_error("Batch must be in1..16");
    std::ifstream file(path,std::ios::binary);
    if(!file) throw std::runtime_error("Cannot open TensorRT plan");
    std::vector<char> bytes((std::istreambuf_iterator<char>(file)),{});
    initLibNvInferPlugins(&logger,"");
    runtime.reset(nvinfer1::createInferRuntime(logger));
    if(!runtime) throw std::runtime_error("Cannot create TensorRT runtime");
    engine.reset(runtime->deserializeCudaEngine(bytes.data(),bytes.size()));
    if(!engine || engine->getNbIOTensors()!=6) throw std::runtime_error("Cannot load expected six-tensor engine");
    if(engine->getTensorDataType("tokens")!=nvinfer1::DataType::kINT32 ||
       engine->getTensorDataType("moves")!=nvinfer1::DataType::kINT32 ||
       engine->getTensorDataType("meta")!=nvinfer1::DataType::kFLOAT)
      throw std::runtime_error("Unexpected input types");
    for(int n=1;n<=maximum;++n) {
      auto slot=std::make_unique<Slot>(*engine,n);
      for(int k=0;k<3;++k) slot->enqueue(stream);
      checked(cudaStreamSynchronize(stream));
      if(graphs) {
        checked(cudaStreamBeginCapture(stream,cudaStreamCaptureModeThreadLocal));
        try {
          slot->enqueue(stream);
          checked(cudaStreamEndCapture(stream,&slot->graph));
          checked(cudaGraphInstantiate(&slot->executable,slot->graph,0));
        } catch(...) {
          cudaStreamCaptureStatus state;
          if(cudaStreamIsCapturing(stream,&state)==cudaSuccess && state!=cudaStreamCaptureStatusNone) {
            cudaGraph_t abandoned=nullptr;cudaStreamEndCapture(stream,&abandoned);
            if(abandoned)cudaGraphDestroy(abandoned);
          }
          throw;
        }
      }
      slots.push_back(std::move(slot));
    }
  }
  ~Impl() {
    cudaStreamSynchronize(stream);
    slots.clear(); engine.reset(); runtime.reset();
  }
};
TrtInference::TrtInference(const std::string &path,int maximum,bool graphs)
    :impl(std::make_unique<Impl>(path,maximum,graphs)) {}
TrtInference::~TrtInference()=default;
int TrtInference::maximum_batch() const { return impl->slots.size(); }
const float *TrtInference::evaluate(const uint8_t *t,const float *m,const int16_t *ids,int n) {
  if(n<1 || n>maximum_batch() || !t || !m || !ids) throw std::runtime_error("Invalid inference batch");
  auto &s=*impl->slots[n-1];
  for(int i=0;i<n*64;++i) s.ht.ptr[i]=t[i];
  for(int i=0;i<n*256;++i) s.hi.ptr[i]=ids[i];
  std::memcpy(s.hm.ptr,m,n*12*sizeof(float));
  if(impl->graphs) checked(cudaGraphLaunch(s.executable,impl->stream));
  else s.enqueue(impl->stream);
  checked(cudaStreamSynchronize(impl->stream)); impl->last=n;
  return s.ho.ptr;
}
const float *TrtInference::raw_output(int n) {
  if(n!=impl->last || n<1) throw std::runtime_error("No matching completed inference");
  auto &s=*impl->slots[n-1];
  checked(cudaMemcpyAsync(s.hr.ptr,s.raw.ptr,n*515*sizeof(float),cudaMemcpyDeviceToHost,impl->stream));
  checked(cudaStreamSynchronize(impl->stream)); return s.hr.ptr;
}
} // namespace nc_native

namespace { thread_local std::string inference_error; }
extern "C" {
void *nci_create(const char *plan,int batch,int graphs) {
  try { return new nc_native::TrtInference(plan,batch,graphs!=0); }
  catch(const std::exception &e) { inference_error=e.what(); return nullptr; }
}
void nci_destroy(void *handle) { delete static_cast<nc_native::TrtInference *>(handle); }
const char *nci_error() { return inference_error.c_str(); }
int nci_fuse_device(const float *p,const float *q,const float *w,const int32_t *ids,
                   float *out,float *raw,int n,uintptr_t stream) {
  try {
    if(n<1 || !p || !q || !w || !ids || !out || !raw) throw std::runtime_error("Invalid head fusion inputs");
    fuse_heads<<<n,256,0,reinterpret_cast<cudaStream_t>(stream)>>>(p,q,w,ids,out,raw);
    checked(cudaGetLastError());inference_error.clear();return 0;
  } catch(const std::exception &e) {inference_error=e.what();return -1;}
}
int nci_evaluate(void *h,const uint8_t *t,const float *m,const int16_t *ids,int n,float *fused,float *raw) {
  try {
    if(!h || !fused) throw std::runtime_error("Invalid handle/output");
    auto &net=*static_cast<nc_native::TrtInference *>(h);
    std::memcpy(fused,net.evaluate(t,m,ids,n),n*515*sizeof(float));
    if(raw) std::memcpy(raw,net.raw_output(n),n*515*sizeof(float));
    inference_error.clear(); return 0;
  } catch(const std::exception &e) { inference_error=e.what(); return -1; }
}
}
