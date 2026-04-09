#include <iostream>
#include <vector>
#include <stdint.h>
#include <cstdlib>

// Descriptors remain the same
struct MemRefDescriptor4D {
    float* allocated;
    float* aligned;
    intptr_t offset;
    intptr_t sizes[4];
    intptr_t strides[4];
};

struct MemRefDescriptor2D {
    float* allocated;
    float* aligned;
    intptr_t offset;
    intptr_t sizes[2];
    intptr_t strides[2];
};

extern "C" {
    // FIX: The returned memref descriptor is passed as the FIRST argument
    void _mlir_ciface_forward(MemRefDescriptor2D* output, MemRefDescriptor4D* input);
}

int main() {
    // Prepare input data
    std::vector<float> input_data(1 * 3 * 32 * 32, 0.1f);

    MemRefDescriptor4D input_memref = {
        input_data.data(), input_data.data(), 0, {1, 3, 32, 32}, {3072, 1024, 32, 1}
    };
    
    // FIX: Leave output uninitialized; MLIR will allocate and fill these pointers
    MemRefDescriptor2D output_memref;

    // FIX: Swap the order of arguments (Output first, Input second)
    _mlir_ciface_forward(&output_memref, &input_memref);

    // FIX: Read from the aligned pointer populated by MLIR
    std::cout << "Inference successful. First output: " << output_memref.aligned[0] << std::endl;

    // Optional: free the memory allocated by the MLIR runtime
    free(output_memref.allocated);

    return 0;
}