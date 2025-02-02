#ifndef IMAGE_HPP
#define IMAGE_HPP

#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <iostream>


// device enum
enum Device{
	CPU = 0,
	GPU = 1
};

// this is our main Image definition 
class Image{
public:
	int ncols, nrows;
	std::vector<float> data;
	float* d_data = nullptr;
	Device device;

	// // zero constructor 
	// Image(int h, int w, Device device = CPU): 
	// 	nrows(h),
	// 	ncols(w),
	// 	data(h*w, 0.0f),
	// 	device(device){
	// 		// to GPU logic if needed

	// }

	// Allocate memory only constructor
	Image(int h, int w, Device device = CPU): 
		nrows(h),
		ncols(w),
		device(device){

		if (device == CPU){
			data.resize(h*w);
		}//else{
	//		cudaMalloc(&d_data, h*w*sizeof(float));
//		}
	}

	// load from .npy constructor
	Image(const std::string& filename): 
		d_data(nullptr){

		load_npy(filename); 
		device = CPU;
	}


	// accessor
	float& operator()(int x, int y){
		return data[x*ncols + y];
	}
	const float& operator()(int x, int y) const{
		return data[x*ncols + y];
	}
	float& at(int x, int y){
		return data[x*ncols + y];
	}
	const float& at(int x, int y) const{
		return data[x*ncols + y];
	}

	// save as .npy
	void save_npy(const std::string& filename) const{
		std::ofstream file(filename, std::ios::binary);

		if (!file){
			throw std::runtime_error("Could not open file " + filename);
		}

		// write .npy header
		const std::string header = make_npy_header(nrows, ncols);
		file.write(header.data(), header.size());
		
		// write data
		file.write(reinterpret_cast<const char*>(data.data()), 
				   data.size()*sizeof(float));
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

		// make room for nrows*ncols floats
		data.resize(nrows*ncols);

		// read data
		if (num_bytes==4){
			// read binary data directly into data
			file.read(reinterpret_cast<char*>(data.data()), data.size()*sizeof(float));
		}else{
			// read binary data in a temporary double buffer
			std::vector<double> temp_data(nrows*ncols);
			file.read(reinterpret_cast<char*>(temp_data.data()), temp_data.size()*sizeof(double));

			// convert to float
			for (size_t i=0; i<data.size(); i++){
				data[i] = static_cast<float>(temp_data[i]);
			}
		}

		if (little_endianness && !is_system_little_endian()){
			swap_endianness(data);  
		}
	}

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

    static std::string make_npy_header(int rows, int cols) {
        std::ostringstream header;
        header << "{'descr': '<f4', 'fortran_order': False, 'shape': (" << rows << ", " << cols << "), }";
        
		// Assuming 10 bytes for header metadata. 
		// We will then pad until the header is 64 bytes aligned 
		while ((header.str().size() + 10) % 64 != 0) header << " ";  

        std::string result = "\x93NUMPY\x01\x00";
        uint16_t header_len = static_cast<uint16_t>(header.str().size());
        result.append(reinterpret_cast<const char*>(&header_len), 2);
        result.append(header.str());

        return result;
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
