//
// $Id$ :$Date$
//

/* 
  Functions which read/write Interfile images
  Pretty horrible implementations at the moment...

  First version: Kris Thielemans
   */

#include "interfile.h"
#include "utilities.h"

PETImageOfVolume read_interfile_image(fstream& input)
{
  KeyParser kp(&input);
  if(kp.StartParsing())
    {
      PETerror("\nError parsing interfile header, \n\
               I am going to ask you lots of questions...\n");
      return ask_image_details();
    }
  
  
  printf("byte order : %s_endian\n",
	 kp.file_byte_order == ByteOrder::little_endian ? "Little" : "Big");
  printf("Number type:  %d\n", kp.type_of_numbers);
  /*
    printf("num_dimensions : %d\n",kp.num_dimensions);
    for(int i=0;i<kp.num_dimensions;i++)
    {
    vector<int> v;
    v=kp.matrix_size[i];
    for(int ii=0;ii<v.size();ii++)
    printf("matrix_size[%d][%d] : %d\n",i+1,ii+1,v[ii]);
    //printf("matrix_labels[%d] : %s\n",i+1, kp.matrix_labels[i].c_str());
    }
  */
  assert(kp.num_dimensions == 3);
  assert(kp.matrix_size[0].size() == 1);
  assert(kp.matrix_size[1].size() == 1);
  assert(kp.matrix_size[2].size() == 1);

  Point3D origin(0,0,0);

  Point3D voxel_size(kp.pixel_sizes[0], kp.pixel_sizes[1], kp.pixel_sizes[2]);

  PETImageOfVolume 
    image(Tensor3D<float>(kp.matrix_size[2][0], kp.matrix_size[1][0], kp.matrix_size[0][0]),
	  origin,
	  voxel_size);
  Real scale = Real(1);
  image.read_data(*(kp.in_stream), kp.type_of_numbers, scale, kp.file_byte_order);
  assert(scale == 1);    

  /*
    // look at the KeyParser class for retrieving 
    // values for the PETSinog. constructor.
    //PETSinogramOfVolume sino;
    vector<int> min_r(9,0);
    
    sino(scanmodel,min_r,kp.matrix_size[max_r_index],ecc);
  */
    
  if(kp.in_stream!=0)
    {
      kp.in_stream->close();
      delete kp.in_stream;
    }
  
  
  return image;
}


template <class NUMBER>
bool write_basic_interfile(const char * const filename, Tensor3D<NUMBER>& image)
{
  string header_name = filename;
  header_name += ".hv";
  ofstream output_header(header_name.c_str(), ios::out);
  if (!output_header.good())
  {
    cerr << "Error opening Interfile header '" 
         << header_name << " for writing" << endl;
    return false;
  }  
  string data_name = filename;
  data_name += ".v";
  ofstream output_data;
  open_write_binary(output_data, data_name.c_str());

  output_header << "!INTERFILE  :=\n";
  output_header << "name of data file := " << data_name << endl;
  output_header << "!GENERAL DATA :=\n";
  output_header << "!GENERAL IMAGE DATA :=\n";
  output_header << "!type of data := PET\n";
  output_header << "imagedata byte order := " <<
    (ByteOrder::get_native_order() == ByteOrder::little_endian 
     ? "LITTLEENDIAN"
     : "BIGENDIAN")
    << endl;
  output_header << "!PET STUDY (General) :=\n";
  output_header << "!PET data type := Image\n";
  output_header << "process status := Reconstructed\n";
  output_header << "!number format := ";
  if (NumericInfo<NUMBER>().integer_type() )
    output_header << ( NumericInfo<NUMBER>().signed_type() 
                      ? "signed integer\n" : "unsigned integer\n");
  else
    output_header << "float\n";
  output_header << "!number of bytes per pixel := " 
                 << NumericInfo<NUMBER>().size_in_bytes() << endl;
  output_header << "number of dimensions := 3\n";
  output_header << "!matrix size [1] := "
                << image.get_length1() << endl;
  output_header << "matrix axis label [1] := x\n";
  output_header << "!matrix size [2] := "
                << image.get_length2() << endl;
  output_header << "matrix axis label [2] := y\n";
  output_header << "!matrix size [3] := "
                << image.get_length3() << endl;
  output_header << "matrix axis label [3] := z\n";
  output_header << "number of time frames := 1\n";
  output_header << "!END OF INTERFILE :=\n";

  image.write_data(output_data);

  // temporary copy to make an old-style header to satisfy Analyze
  {
    string header_name = filename;
    header_name += ".ahv";
    ofstream output_header(header_name.c_str(), ios::out);
    if (!output_header.good())
    {
      cerr << "Error opening old-style Interfile header '" 
	<< header_name << " for writing" << endl;
      return false;
    }  
    string data_name = filename;
    // point to the same file with binary data
    data_name += ".v";
    ofstream output_data;
    open_write_binary(output_data, data_name.c_str());
    
    output_header << "!INTERFILE  :=\n";
    output_header << "!name of data file := " << data_name << endl;
    output_header << "!total number of images := "
      << image.get_length3() << endl;
    output_header << "imagedata byte order := " <<
      (ByteOrder::get_native_order() == ByteOrder::little_endian 
      ? "LITTLEENDIAN"
      : "BIGENDIAN")
      << endl;
    output_header << "!number format := ";
    if (NumericInfo<NUMBER>().integer_type() )
      output_header << ( NumericInfo<NUMBER>().signed_type() 
      ? "signed integer\n" : "unsigned integer\n");
    else
      output_header << "float\n";
    output_header << "!number of bytes per pixel := " 
      << NumericInfo<NUMBER>().size_in_bytes() << endl;
    output_header << "!matrix size [1] := "
      << image.get_length1() << endl;
    output_header << "matrix axis label [1] := x\n";
    output_header << "!matrix size [2] := "
      << image.get_length2() << endl;
    output_header << "matrix axis label [2] := y\n";
    output_header << "!END OF INTERFILE :=\n";
    
  }
  return true;
}

/**********************************************************************
 template instantiations
 **********************************************************************/

// Codewarrior gets confused by the syntax of template instantiation
#ifndef __MSL__
template bool write_basic_interfile(const char * const filename, Tensor3D<signed char>& );
template bool write_basic_interfile(const char * const filename, Tensor3D<unsigned char>&);
template bool write_basic_interfile(const char * const filename, Tensor3D<signed short>& image);
template bool write_basic_interfile(const char * const filename, Tensor3D<unsigned short>&);
template bool write_basic_interfile(const char * const filename, Tensor3D<float>& image);
#else
template bool write_basic_interfile<>(const char * const filename, Tensor3D<signed char>& );
template bool write_basic_interfile<>(const char * const filename, Tensor3D<unsigned char>&);
template bool write_basic_interfile<>(const char * const filename, Tensor3D<signed short>& image);
template bool write_basic_interfile<>(const char * const filename, Tensor3D<unsigned short>&);
template bool write_basic_interfile<>(const char * const filename, Tensor3D<float>& image);
#endif
