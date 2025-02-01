#ifndef IMAGE_H
#define IMAGE_H

#include <vector>
#include <fstream>
#include <sstream>
#include <string>
#include <algorithm>
#include <iostream>


// this is our main Image definition 
class Image{
public:
	int ncols, nrows;
	std::vector<float> data;

	// zero constructor 
	Image(int h, int w): nrows(h),
						 ncols(w),
			     		 data(h*w, 0.0f){}

	// load from .npy constructor
	Image(const std::string& filename){
		load_npy(filename);
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
  
		// extract shape (assuming a 2d float32 array)
		bool little_endianness;
		parse_npy_header(header, little_endianness);	

		// read binary data
        data.resize(nrows*ncols);
        file.read(reinterpret_cast<char*>(data.data()), data.size()*sizeof(float));

		std::cout << data[0] << std::endl;
		std::cout << data[1] << std::endl;
		std::cout << data[2] << std::endl;
		std::cout << data[3] << std::endl;
		std::cout << data[4] << std::endl;
		std::cout << data[5] << std::endl;


		if (little_endianness && !is_system_little_endian()){
			swap_endianness(data);  
		}
	}

	void parse_npy_header(const std::string& header,
						  bool& little_endianness){
		// Example header: "{ 'descr': '<f4', 'fortran_order': False, 'shape': (128, 256), }"

		// we sum 10 here in order to get the position of the first digit of the shape
		std::size_t shape_pos = header.find("'shape': (") + 10;
		if (shape_pos == std::string::npos){
			throw std::runtime_error("Invalid .npy header format. Could not find shape.");
		}
		std::stringstream shape_stream(header.substr(shape_pos));
		char ignore;
		shape_stream >> nrows >> ignore >> ncols;

        little_endianness = (header.find("<i8") != std::string::npos);

	}

    static std::string make_npy_header(int rows, int cols) {
        std::ostringstream header;
        header << "{'descr': '<i8', 'fortran_order': False, 'shape': (" << rows << ", " << cols << "), }";
        
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
