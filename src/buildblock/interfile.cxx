//
// $Id$ :$Date$
//

/* 
  Functions which read/write Interfile images
  Pretty horrible implementations at the moment...

  First version: Kris Thielemans
   */

// KT 19/10/98 use new class
#include "InterfileHeader.h"

#include "interfile.h"
#include "utilities.h"

// KT 14/10/98 make arg istream
PETImageOfVolume read_interfile_image(istream& input)
{
  // KT 19/10/98 use new class
  InterfileImageHeader kp(input);
  if (!kp.parse())
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
    // TODO move to ImageHeader post_processing
    assert(kp.num_dimensions == 3);
    assert(kp.matrix_size[0].size() == 1);
    assert(kp.matrix_size[1].size() == 1);
    assert(kp.matrix_size[2].size() == 1);

    //KT 26/10/98 now here
    ifstream data_in;
    open_read_binary(data_in, kp.data_file_name.c_str());

    Point3D origin(0,0,0);

    Point3D voxel_size(kp.pixel_sizes[0], kp.pixel_sizes[1], kp.pixel_sizes[2]);

    //TODO adjust x,y ranges to centre ?
    PETImageOfVolume 
     image(Tensor3D<float>(kp.matrix_size[2][0], kp.matrix_size[1][0], kp.matrix_size[0][0]),
		      origin,
		      voxel_size);

    //KT 26/10/98 do file offset now
    data_in.seekg(kp.data_offset[0]);

    Real scale = Real(1);
    //KT 26/10/98 use local stream data_in
    image.read_data(data_in, kp.type_of_numbers, scale, kp.file_byte_order);
    assert(scale == 1);  
    
    //KT 26/10/98 take scale factor into account
    if (kp.image_scaling_factor[0]!= 1)
      image *= kp.image_scaling_factor[0];

    //KT 26/10/98 not necessary anymore
    /*
    if(kp.in_stream!=0)
      {
	kp.in_stream->close();
	delete kp.in_stream;
      }
      */
  
  return image;
}


template <class NUMBER>
bool write_basic_interfile(const char * const filename, const Tensor3D<NUMBER>& image)
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
    // KT 13/10/98 point to same binary data as above
    data_name += ".v";
    
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

PETSinogramOfVolume read_interfile_PSOV(istream& input)
{
  //KT 26/10/98 new
  // Horrible trick to keep the stream alive. At the moment, this pointer
  // is never deleted
  // TODO
  fstream *data_in =  new fstream;
  

  InterfilePSOVHeader kp(input);
  if (!kp.parse())
    {
      PETerror("\nError parsing interfile header, \n\
     I am going to ask you lots of questions...\n");
      return ask_PSOV_details(data_in);
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


    open_read_binary(*data_in, kp.data_file_name.c_str());

    vector<int> min_ring_num_per_segment(kp.num_segments);
    vector<int> max_ring_num_per_segment(kp.num_segments);
    for (int s=0; s<kp.num_segments; s++)
    {
      min_ring_num_per_segment[s] = 0;
      max_ring_num_per_segment[s] = kp.num_rings_per_segment[s]-1;
    }

    // TODO Horrible trick to get the scanner right
    PETScannerInfo scanner;
    switch (kp.num_views)
    {
    case 96:
      scanner = PETScannerInfo::RPT;
      break;
    case 192:
      scanner = PETScannerInfo::E953;
      break;
    case 336:
      scanner = PETScannerInfo::Advance;
      break;
    case 288:
      scanner = PETScannerInfo::E966;
      break;
    default:
      cerr << "Interfile warning: I did not recognise the scanner from num_views.\n\
Defaulting to RPT" << endl;
      scanner = PETScannerInfo::RPT;
      break;
    }
    return PETSinogramOfVolume(
                      scanner,
		      kp.segment_sequence,
		      kp.min_ring_difference, 
		      kp.max_ring_difference, 
		      min_ring_num_per_segment,
		      max_ring_num_per_segment,
		      0, kp.num_views-1,
		      -kp.num_bins/2,
		      (-kp.num_bins/2) + kp.num_bins-1,
		      *data_in,
		      kp.data_offset[0],
		      kp.storage_order,
		      kp.type_of_numbers,
		      kp.image_scaling_factor[0]);
}


/**********************************************************************
 template instantiations
 **********************************************************************/

// Codewarrior gets confused by the syntax of template instantiation
#ifndef __MSL__
template bool write_basic_interfile(const char * const filename, const Tensor3D<signed char>& );
template bool write_basic_interfile(const char * const filename, const Tensor3D<unsigned char>&);
template bool write_basic_interfile(const char * const filename, const Tensor3D<signed short>&);
template bool write_basic_interfile(const char * const filename, const Tensor3D<unsigned short>&);
template bool write_basic_interfile(const char * const filename, const Tensor3D<float>&);
#else
template bool write_basic_interfile<>(const char * const filename, const Tensor3D<signed char>& );
template bool write_basic_interfile<>(const char * const filename, const Tensor3D<unsigned char>&);
template bool write_basic_interfile<>(const char * const filename, const Tensor3D<signed short>&);
template bool write_basic_interfile<>(const char * const filename, const Tensor3D<unsigned short>&);
template bool write_basic_interfile<>(const char * const filename, const Tensor3D<float>&);
#endif
