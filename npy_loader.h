#ifndef NPY_LOADER_H
#define NPY_LOADER_H

#include <vector>
#include <string>
#include <fstream>
#include <cstdint>
#include <stdexcept>

// Simple .npy file loader for float32 and int8 arrays
// Supports only little-endian, C-contiguous (row-major) arrays

struct NpyArray {
    std::vector<size_t> shape;
    std::vector<float> data_float;
    std::vector<int8_t> data_int8;
    bool is_float;
    
    size_t size() const {
        size_t s = 1;
        for (size_t dim : shape) s *= dim;
        return s;
    }
};

inline NpyArray load_npy(const std::string& filename) {
    std::ifstream file(filename, std::ios::binary);
    if (!file.is_open()) {
        throw std::runtime_error("Cannot open file: " + filename);
    }
    
    // Read magic string
    char magic[6];
    file.read(magic, 6);
    if (magic[0] != '\x93' || magic[1] != 'N' || magic[2] != 'U' || 
        magic[3] != 'M' || magic[4] != 'P' || magic[5] != 'Y') {
        throw std::runtime_error("Invalid .npy file: " + filename);
    }
    
    // Read version (uint8 major, uint8 minor)
    uint8_t major, minor;
    file.read(reinterpret_cast<char*>(&major), 1);
    file.read(reinterpret_cast<char*>(&minor), 1);
    
    // Read header length (uint16 for v1.0, uint32 for v2.0+)
    uint16_t header_len;
    if (major == 1) {
        file.read(reinterpret_cast<char*>(&header_len), 2);
    } else {
        uint32_t header_len_32;
        file.read(reinterpret_cast<char*>(&header_len_32), 4);
        header_len = static_cast<uint16_t>(header_len_32);
    }
    
    // Read header
    std::string header(header_len, '\0');
    file.read(&header[0], header_len);
    
    // Parse header to extract shape and dtype
    std::vector<size_t> shape;
    std::string dtype;
    bool fortran_order = false;
    
    // Simple parsing (assumes standard format)
    size_t shape_start = header.find("'shape':");
    size_t dtype_start = header.find("'descr':");
    size_t fortran_start = header.find("'fortran_order':");
    
    if (shape_start == std::string::npos || dtype_start == std::string::npos) {
        throw std::runtime_error("Cannot parse .npy header");
    }
    
    // Extract dtype (e.g., '<f4' for float32, '<i1' for int8)
    size_t dtype_val_start = header.find("'", dtype_start + 8) + 1;
    size_t dtype_val_end = header.find("'", dtype_val_start);
    dtype = header.substr(dtype_val_start, dtype_val_end - dtype_val_start);
    
    // Extract shape
    size_t shape_open = header.find("(", shape_start);
    size_t shape_close = header.find(")", shape_open);
    std::string shape_str = header.substr(shape_open + 1, shape_close - shape_open - 1);
    
    // Parse shape tuple
    size_t pos = 0;
    while (pos < shape_str.length()) {
        size_t comma = shape_str.find(',', pos);
        if (comma == std::string::npos) comma = shape_str.length();
        std::string dim_str = shape_str.substr(pos, comma - pos);
        // Remove whitespace
        dim_str.erase(0, dim_str.find_first_not_of(" \t"));
        dim_str.erase(dim_str.find_last_not_of(" \t") + 1);
        if (!dim_str.empty()) {
            shape.push_back(std::stoull(dim_str));
        }
        pos = comma + 1;
    }
    
    // Extract fortran_order
    if (fortran_start != std::string::npos) {
        size_t fortran_val_start = header.find_first_of("TF", fortran_start);
        if (fortran_val_start != std::string::npos) {
            fortran_order = (header[fortran_val_start] == 'T');
        }
    }
    
    // Calculate data size
    size_t total_size = 1;
    for (size_t dim : shape) total_size *= dim;
    
    NpyArray arr;
    arr.shape = shape;
    
    // Read data based on dtype
    // Handle different dtype formats: <f4, f4, |f4 for float32; <i1, i1, |i1 for int8
    bool is_float32 = (dtype == "<f4" || dtype == "f4" || dtype == "|f4" || 
                       dtype.find("f4") != std::string::npos);
    bool is_int8 = (dtype == "<i1" || dtype == "i1" || dtype == "|i1" || 
                    dtype.find("i1") != std::string::npos);
    
    if (is_float32) {
        arr.is_float = true;
        arr.data_float.resize(total_size);
        file.read(reinterpret_cast<char*>(arr.data_float.data()), 
                  total_size * sizeof(float));
    } else if (is_int8) {
        arr.is_float = false;
        arr.data_int8.resize(total_size);
        file.read(reinterpret_cast<char*>(arr.data_int8.data()), 
                  total_size * sizeof(int8_t));
    } else {
        throw std::runtime_error("Unsupported dtype: " + dtype);
    }
    
    // Convert Fortran-order (column-major) to C-order (row-major) if needed
    if (fortran_order && shape.size() == 2) {
        size_t rows = shape[0];
        size_t cols = shape[1];
        
        if (arr.is_float) {
            std::vector<float> temp(total_size);
            // Transpose: F-order to C-order
            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < cols; ++j) {
                    // F-order: data[i + j*rows]
                    // C-order: data[i*cols + j]
                    temp[i * cols + j] = arr.data_float[i + j * rows];
                }
            }
            arr.data_float = std::move(temp);
        } else {
            std::vector<int8_t> temp(total_size);
            for (size_t i = 0; i < rows; ++i) {
                for (size_t j = 0; j < cols; ++j) {
                    temp[i * cols + j] = arr.data_int8[i + j * rows];
                }
            }
            arr.data_int8 = std::move(temp);
        }
    }
    
    file.close();
    return arr;
}

#endif // NPY_LOADER_H
