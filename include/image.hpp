#ifndef IMAGE_H
#define IMAGE_H

#include <vector>
#include <fstream>
#include <sstream>
#include <string>


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


	// accessor
	float& operator()(int x, int y){
		return data[y*ncols + x];
	}
	const float& operator()(int x, int y) const{
		return data[y*ncols + x];
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
		parse_npy_header(header);	
	}

	void parse_npy_header(const std::string& header){
		// Example header: "{ 'descr': '<f4', 'fortran_order': False, 'shape': (128, 256), }"

		// we sum 10 here in order to get the position of the first digit of the shape
		std::size_t shape_pos = header.find("'shape': (") + 10;
		if (shape_pos == std::string::npos){
			throw std::runtime_error("Invalid .npy header format. Could not find shape.");
		}

		std::stringstream shape_stream(header.substr(shape_pos));
		char ignore;
		shape_stream >> nrows >> ignore >> ncols;
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
};

#endif
