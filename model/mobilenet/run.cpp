#include <iostream>
#include <vector>
#include "build/weights.h" 

int main() {
    std::cout << "Allocating memory...\n";
    
    std::vector<float> inputData(1 * 3 * 32 * 32, 1.0f);
    MemRefDescriptor<float, 4> inputMemRef = {
        inputData.data(), inputData.data(), 0, 
        {1, 3, 32, 32}, {3072, 1024, 32, 1}
    };

    std::vector<float> outputData(10, 0.0f);
    MemRefDescriptor<float, 2> outputMemRef = {
        outputData.data(), outputData.data(), 0, 
        {1, 10}, {10, 1}
    };

    std::cout << "Loading weights from build/weights.bin...\n";
    std::ifstream weight_file("build/weights.bin", std::ios::binary);
    if (!weight_file) {
        std::cerr << "Failed to open build/weights.bin!\n";
        return 1;
    }

    std::vector<void*> weight_memrefs;
    std::vector<std::vector<float>*> weight_vectors;
    load_weights(weight_file, weight_memrefs, weight_vectors);

    std::cout << "Running mobilenet...\n";
    run_forward(&outputMemRef, &inputMemRef, weight_memrefs);
    std::cout << "Logits: ";
    float* result_ptr = outputMemRef.aligned; 
    
    for (int i = 0; i < 10; ++i) {
        std::cout << result_ptr[i] << " ";
    }
    std::cout << "\n";

    free(outputMemRef.allocated);
    return 0;
}