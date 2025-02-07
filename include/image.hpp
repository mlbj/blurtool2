#ifndef IMAGE_HPP
#define IMAGE_HPP

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <iostream>


#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>


// this is our main Image definition 
class Image{
public:
	int ncols, nrows;
	std::vector<float> data;
	float* d_data = nullptr;

	// Allocate memory only constructor
	Image(int h, int w, Device device = CPU): 
		nrows(h),
		ncols(w),
		device(device){
		cudaMalloc(&d_data, h*w*sizeof(float));
		cudaMemset(d_data, 0, h*w*sizeof(float));
	}

	// load from .npy constructor
	Image(const std::string& filename): 
		d_data(nullptr){

		load_npy(filename); 
		device = GPU;
	}


	// // accessor in CPU
	// float& operator()(int x, int y){
	// 	if (device==GPU){
	// 		throw std::runtime_error("Cannot return GPU reference. Check device attribute.");
	// 	}else{
	// 		return data[x*ncols + y];
	// 	}
	// }
	// const float& operator()(int x, int y) const{
	// 	if (device==GPU){
	// 		throw std::runtime_error("Cannot return GPU reference. Check device attribute.");
	// 	}else{
	// 		return data[x*ncols + y];
	// 	}
	// }
	// float& at(int x, int y){
	// 	if (device==GPU){
	// 		throw std::runtime_error("Cannot return GPU reference. Check device attribute");
	// 	}else{
	// 		return data[x*ncols + y];
	// 	}
	// }
	// const float& at(int x, int y) const{
	// 	if (device==GPU){
	// 		throw std::runtime_error("Cannot return GPU reference. Check device attribute.");
	// 	}else{
	// 		return data[x*ncols + y];
	// 	}
	// }

	// accessor-like device functions for the GPU case
	__device__ float gpu_at(int x, int y);
	__device__ void set_gpu_value(int x, int y, float value);

	// save as .npy
	void save_npy(const std::string& filename) const{
		// save device data to host 
		std::vector<float> h_data(nrows*ncols);
		cudaMemcpy(h_data.data(), d_data, nrows*ncols*sizeof(float), cudaMemcpyDeviceToHost);
		
		std::ofstream file(filename, std::ios::binary);

		if (!file){
			throw std::runtime_error("Could not open file " + filename);
		}

		// write .npy header
		const std::string header = make_npy_header(nrows, ncols);
		file.write(header.data(), header.size());
		
		// write data
		file.write(reinterpret_cast<const char*>(h_data.data()), 
				   h_data.size()*sizeof(float));
	}

	~Image(){
		if (d_data){
			cudaFree(d_data);
		}
	}


private: 
	void load_npy(const std::string& filename){ 
		std::ifstream file(filename, std::ios::binary);
		if (!file){
			throw std::runtime_error("Could not open file " + filename);
		}
		
		// read magic string 
		char magic[6];
		file.read(magic, 6);
		if (std::string(magic, 6) != "\x93NUMPY"){
			throw std::runtime_error("Invalid magic string in file " + filename);
		}

		// skip version 
		file.ignore(2);

		// get header_len
		uint16_t header_len;
		file.read(reinterpret_cast<char*>(&header_len), 2); 

		// read header
		std::string header(header_len, ' ');
		file.read(header.data(), header_len);

		// extract shape (assuming a 2d float array)
		bool little_endianness;
		size_t num_bytes;
		parse_npy_header(header, little_endianness, num_bytes);	

		// allocate host data
		std::vector<float h_data(nrows*ncols);

		// make room for nrows*ncols floats
		data.resize(nrows*ncols);

		// read host data
		if (num_bytes==4){
			// read binary data directly into data
			file.read(reinterpret_cast<char*>(h_data.data()), h_data.size()*sizeof(float));
		}else{
			// read binary data in a temporary double buffer
			std::vector<double> temp_h_data(nrows*ncols);
			file.read(reinterpret_cast<char*>(temp_h_data.data()), temp_h_data.size()*sizeof(double));

			// convert to float
			for (size_t i=0; i<h_data.size(); i++){
				h_data[i] = static_cast<float>(temp_h_data[i]);
			}
		}

		// move data to GPU
		cudaMalloc(&d_data, nrows*ncols*sizeof(float));
		cudaMemcpy(d_data, h_data.data(), nrows*ncols*sizeof(float), cudaMemcpyHostToDevice);
	}


// this


	void parse_npy_header(const std::string& header,
						  bool& little_endianness,
						  size_t& num_bytes){
		// Example header: "{ 'descr': '<f4', 'fortran_order': False, 'shape': (128, 256), }"
		
		// decode shape 
		// we sum 10 here in order to get the position of the first digit of the shape
		std::size_t shape_pos = header.find("'shape': (") + 10;
		if (shape_pos == std::string::npos){
			throw std::runtime_error("Invalid .npy header format. Could not find shape.");
		}
		std::stringstream shape_stream(header.substr(shape_pos));
		char ignore;
		shape_stream >> nrows >> ignore >> ncols;

		// decode data type 
		std::size_t descr_pos = header.find("'descr': '") + 10;
		if (descr_pos == std::string::npos){
			throw std::runtime_error("Invalid .npy header format. Could not find data type.");
		}
		std::string descr = header.substr(descr_pos, 3); 
		if (descr == "<f4"){
			num_bytes = 4;
        	little_endianness = true;
		}else if (descr == "<f8"){ 
			num_bytes = 8;
        	little_endianness = true;
		}else{
			throw std::runtime_error("Invalid .npy header format. Unsupported data type.");
		}

	}

	static std::string make_npy_header(int rows, int cols){
    	std::ostringstream header;
    	header << "{'descr': '<f4', 'fortran_order': False, 'shape': (" << rows << ", " << cols << "), }";

    	// padding to align to 64 bytes
    	while ((header.str().size() + 10) % 64 != 0){
			header << " ";  
		}

		// create result string as a vector of chars
    	std::vector<char> result;
    
    	// numpy magic string
		// we will exclude null terminator of C-string because
		// it was causing some trouble 
    	const char magic[] = "\x93NUMPY\x01\x00"; 
    	result.insert(result.end(), std::begin(magic), std::end(magic) - 1);  

		// get header len 
		uint16_t header_len = static_cast<uint16_t>(header.str().size());
    
    	// append the header length in little-endian format
    	result.push_back(static_cast<char>(header_len & 0xFF));        // Low byte
    	result.push_back(static_cast<char>((header_len >> 8) & 0xFF)); // High byte
    
    	// append the actual header
	    std::string header_str = header.str();
	    result.insert(result.end(), header_str.begin(), header_str.end());

		// convert vector to string
    	return std::string(result.begin(), result.end());  
	}	

    // check if system is little-endian
    static bool is_system_little_endian() {
        uint16_t num = 1;
        return reinterpret_cast<uint8_t*>(&num)[0] == 1;
    }

    // swap endianness of float vector
    static void swap_endianness(std::vector<float>& vec) {
        for (float& value : vec) {
            char* bytes = reinterpret_cast<char*>(&value);
            std::reverse(bytes, bytes + sizeof(float));
        }
    }
};

#endif
