#ifndef IMAGE_HPP
#define IMAGE_HPP

#include <cuda_runtime.h>
#include <cufft.h>
#include <stdexcept>
#include <string>
#include <iostream>


#include <vector>
#include <fstream>
#include <sstream>
#include <algorithm>
 
enum Device{
	CPU=0,
	GPU=1,
};

// this is our main Image definition 
class Image{
public:
	int ncols, nrows;
	Device device; 
	cufftComplex* d_data = nullptr;
	cufftComplex* h_data = nullptr; 

	// Allocate memory only constructor
	Image(int h, int w, Device device = CPU): 
		nrows(h), ncols(w), device(device){
		
		if (device==GPU){
			cudaMalloc(&d_data, h*w*sizeof(cufftComplex));
			cudaMemset(d_data, 0, h*w*sizeof(cufftComplex));
		}else{
			h_data = new cufftComplex[nrows*ncols]();
		}
	}

	// load from .npy constructor
	Image(const std::string& filename, Device device = CPU){
		// load npy from file. If already allocated, this will overwrite host data
		load_npy(filename); 

		// transfer host data to GPU, if necessary 
		if (device == GPU){
			to_gpu();
			
			// free host data
			free_h_data();
		}
	}
        

	// accessor in CPU
	cufftComplex& operator()(int x, int y){
	 	if (device==GPU){
	 		throw std::runtime_error("Cannot return GPU reference. Check device attribute.");
	 	}else{
	 		return h_data[x*ncols + y];
	 	}
	 }
	 const cufftComplex& operator()(int x, int y) const{
	 	if (device==GPU){
	 		throw std::runtime_error("Cannot return GPU reference. Check device attribute.");
	 	}else{
	 		return h_data[x*ncols + y];
	 	}
	 }
	 cufftComplex& at(int x, int y){
	 	if (device==GPU){
	 		throw std::runtime_error("Cannot return GPU reference. Check device attribute");
	 	}else{
	 		return h_data[x*ncols + y];
	 	}
	 }
	 const cufftComplex& at(int x, int y) const{
	 	if (device==GPU){
	 		throw std::runtime_error("Cannot return GPU reference. Check device attribute.");
	 	}else{
	 		return h_data[x*ncols + y];
	 	}
	 }
	

	// accessor-like device functions for the GPU case
	__device__ cufftComplex gpu_at(int x, int y);
	__device__ void set_gpu_value(int x, int y, cufftComplex value);
	
	
	// automatic transfer functions between GPU and CPU
	void to_gpu(){
		if (h_data){
			if (!d_data){
				cudaMalloc(&d_data, nrows*ncols*sizeof(cufftComplex));
			}
			cudaMemcpy(d_data, h_data, nrows*ncols*sizeof(cufftComplex), cudaMemcpyHostToDevice);
			
			// set device flag	
			device = GPU;
		}else{
			// throw warning
		}
	}
	void to_cpu(){
		if (d_data){
			if (!h_data){
				h_data = new cufftComplex[nrows*ncols];
			}
			cudaMemcpy(h_data, d_data, nrows*ncols*sizeof(cufftComplex), cudaMemcpyDeviceToHost);
			
			// set device flag
			device = CPU;	
		}else{
			// throw warning
		}
	}

	// save as .npy
	void save_npy(const std::string& filename){
		if (device == GPU){
			to_cpu();
		}
		
		std::ofstream file(filename, std::ios::binary);
		if (!file){
			throw std::runtime_error("Could not open file " + filename);
		}

		// write .npy header
		const std::string header = make_npy_header(nrows, ncols);
		file.write(header.data(), header.size());
		 
		// for now, we are saving only the real part
		// first we create a float array to store the real parts
		float* h_data_real = new float[nrows*ncols];

		// extract the real parts
		for (int i=0; i<nrows; i++){
			for (int j=0; j<ncols; j++){
				h_data_real[i*ncols + j] = h_data[i*ncols + j].x;
			}
		}

		// write data
		file.write(reinterpret_cast<const char*>(h_data_real), nrows*ncols*sizeof(float));

		// clean up
		delete[] h_data_real;

	}

	~Image(){
		free_h_data();
		free_d_data();
	}


private: 
	void free_h_data(){
		if (h_data){
			delete[] h_data;
			h_data = nullptr;
		}
	}
	
	void free_d_data(){
		if (d_data){
			cudaFree(d_data);
			d_data = nullptr;
		}
	}


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

		// extract shape (assuming a 2d cufftComplex array)
		bool little_endianness;
		size_t num_bytes;
		parse_npy_header(header, little_endianness, num_bytes);	

		// Allocate host memory if necessary
		if (!h_data) {
		    h_data = new cufftComplex[nrows*ncols];
		}


		// read host data
		if (num_bytes==4){
			// assuming float precision
			std::vector<float> h_data_float(nrows*ncols);

			// read binary data directly into data
			file.read(reinterpret_cast<char*>(h_data_float.data()), nrows*ncols*sizeof(float));
			
			// convert to cufftComplex
			for (int i=0; i<nrows*ncols; i++){
				h_data[i].x = h_data_float[i];
				h_data[i].y = 0.0f;
			}
 
 		}else if (num_bytes==8){
			// assuming double precision 
			// read binary data in a temporary double buffer
			std::vector<double> h_data_double(nrows*ncols);
			file.read(reinterpret_cast<char*>(h_data_double.data()), nrows*ncols*sizeof(double));

			// convert to cufftComplex
			for (size_t i=0; i<nrows*ncols; i++){
				h_data[i].x = (float) h_data_double[i];
				h_data[i].y = (float) 0.0f;
			}
		}
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

    // swap endianness of cufftComplex vector
    static void swap_endianness(std::vector<cufftComplex>& vec) {
        for (cufftComplex& value : vec) {
            char* bytes = reinterpret_cast<char*>(&value);
            std::reverse(bytes, bytes + sizeof(cufftComplex));
        }
    }
};

#endif

