#include <iostream>
#include <iomanip>
#include <stdint.h>

// Replace separate structs with a clean templated descriptor
template <typename T, int N> struct MemRefDescriptor {
    T *allocated;
    T *aligned;
    intptr_t offset;
    intptr_t sizes[N];
    intptr_t strides[N];
};

extern "C" {
    // The returned memref descriptor is passed as the FIRST argument
    void _mlir_ciface_forward(MemRefDescriptor<float, 2>* output, MemRefDescriptor<float, 4>* input);
}

int main() {
    const int batch_size = 1;
    const int out_features = 13;

    // Use stack-allocated arrays instead of std::vector
    float input_data[batch_size][3][32][32];
    float output_data[batch_size][out_features];

    // Initialize input data
    for (int b = 0; b < batch_size; b++) {
        for (int c = 0; c < 3; c++) {
            for (int h = 0; h < 32; h++) {
                for (int w = 0; w < 32; w++) {
                    input_data[b][c][h][w] = 0.1f;
                }
            }
        }
    }

    // Set up descriptors pointing to the stack-allocated memory
    MemRefDescriptor<float, 4> input_memref = {
        (float *)input_data, (float *)input_data, 0, 
        {batch_size, 3, 32, 32}, {3072, 1024, 32, 1}
    };
    
    MemRefDescriptor<float, 2> output_memref = {
        (float *)output_data, (float *)output_data, 0, 
        {batch_size, out_features}, {out_features, 1}
    };

    // Execute (Output first, Input second)
    _mlir_ciface_forward(&output_memref, &input_memref);

    // Read and print from the aligned pointer
    float *output = (float *)output_memref.aligned;
    for (int64_t i = 0; i < batch_size; ++i) {
        std::cout << "Batch " << i << ": ";
        for (int64_t j = 0; j < out_features; ++j) {
            std::cout << std::fixed << std::setprecision(5) << output[i * out_features + j] << ' ';
        }
        std::cout << "\n";
    }

    // No free() needed since we use stack allocation
    return 0;
}