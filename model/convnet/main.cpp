#include <iomanip>
#include <iostream>
#include <cstdint>

template <typename T, int N> struct MemRefDescriptor {
  T *allocated;
  T *aligned;
  intptr_t offset;
  intptr_t sizes[N];
  intptr_t strides[N];
};

extern "C" {
  void _mlir_ciface_forward(MemRefDescriptor<float, 2> *output,
                            MemRefDescriptor<float, 4> *input);
}

int main() {
  const int batch_size = 1;
  const int out_features = 13;

  float inputData[batch_size][3][32][32];
  float outputData[batch_size][out_features];

  for (int b = 0; b < batch_size; b++) {
    for (int c = 0; c < 3; c++) {
      for (int h = 0; h < 32; h++) {
        for (int w = 0; w < 32; w++) {
          inputData[b][c][h][w] = 1.0f;
        }
      }
    }
  }

  MemRefDescriptor<float, 4> inputMemRef = {
      (float *)inputData, (float *)inputData, 0,
      {batch_size, 3, 32, 32}, {3072, 1024, 32, 1}
  };

  MemRefDescriptor<float, 2> outputMemRef = {
      (float *)outputData, (float *)outputData, 0,
      {batch_size, out_features}, {out_features, 1}
  };

  _mlir_ciface_forward(&outputMemRef, &inputMemRef);

  float *output = (float *)outputMemRef.aligned;
  for (int64_t i = 0; i < batch_size; ++i) {
    std::cout << "Batch " << i << ": ";
    for (int64_t j = 0; j < out_features; ++j) {
      std::cout << std::fixed << std::setprecision(5) 
                << output[i * out_features + j] << ' ';
    }
    std::cout << "\n";
  }

  return 0;
}