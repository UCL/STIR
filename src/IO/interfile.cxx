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
  // KT 13/10/98 moved stream to parse()
  InterfileImageHeader hdr;
  if (!hdr.parse(input))
    {
      PETerror("\nError parsing interfile header, \n\
      I am going to ask you lots of questions...\n");
      return ask_image_details();
    }
  
  
    //KT 26/10/98 now here
    ifstream data_in;
    open_read_binary(data_in, hdr.data_file_name.c_str());

    Point3D origin(0,0,0);

    Point3D voxel_size(hdr.pixel_sizes[0], hdr.pixel_sizes[1], hdr.pixel_sizes[2]);

    // KT 29/10/98 adjusted x,y ranges to centre 
    const y_size =  hdr.matrix_size[1][0];
    const x_size =  hdr.matrix_size[0][0];
    PETImageOfVolume 
      image(Tensor3D<float>(0,hdr.matrix_size[2][0]-1,
      -y_size/2, (-y_size/2) + y_size - 1,
      -x_size/2, (-x_size/2) + x_size - 1), 
      origin,
      voxel_size);

    //KT 26/10/98 do file offset now
    data_in.seekg(hdr.data_offset[0]);

    Real scale = Real(1);
    //KT 26/10/98 use local stream data_in
    image.read_data(data_in, hdr.type_of_numbers, scale, hdr.file_byte_order);
    assert(scale == 1);  
    
    // KT 26/10/98 take scale factor into account
    // KT 29/10/98 use list of factors
    for (int i=0; i< hdr.matrix_size[2][0]; i++)
      if (hdr.image_scaling_factors[0][i]!= 1)
	image[i] *= hdr.image_scaling_factors[0][i];

 
    return image;
}


// KT 13/11/98 new
PETImageOfVolume read_interfile_image(const char *const filename)
{
    ifstream image_stream(filename);
    if (!image_stream)
    { 
      PETerror("read_interfile_image: couldn't open file %s\n", filename);
      exit(1);
    }

  return read_interfile_image(image_stream);
}

// KT 09/11/98 added voxel_size & output_type
template <class NUMBER>
bool write_basic_interfile(const char * const filename, 
			   const Tensor3D<NUMBER>& image,
			   const Point3D& voxel_size,
			   const NumericType output_type)
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

  // KT 09/11/98 use output_type
  output_header << "!number format := ";
  if (output_type.integer_type() )
    output_header << (output_type.signed_type() 
    ? "signed integer\n" : "unsigned integer\n");
  else
    output_header << "float\n";
  output_header << "!number of bytes per pixel := "
    << output_type.size_in_bytes() << endl;
  
  output_header << "number of dimensions := 3\n";
  output_header << "!matrix size [1] := "
                << image.get_length1() << endl;
  output_header << "matrix axis label [1] := x\n";
  output_header << "scaling factor (mm/pixel) [1] := " 
                << voxel_size.x << endl;
  output_header << "!matrix size [2] := "
                << image.get_length2() << endl;
  output_header << "matrix axis label [2] := y\n";
  output_header << "scaling factor (mm/pixel) [2] := " 
                << voxel_size.y << endl;
  output_header << "!matrix size [3] := "
                << image.get_length3() << endl;
  output_header << "matrix axis label [3] := z\n";
  output_header << "scaling factor (mm/pixel) [3] := " 
                << voxel_size.z << endl;
  output_header << "number of time frames := 1\n";

  // KT 09/11/98 use output_type
  Real scale = 0;
  image.write_data(output_data, output_type, scale);
 

  output_header << "image scaling factor[1] := " << scale << endl;
  output_header << "quantification units := " << scale << endl;
  output_header << "maximum pixel count := " << image.find_max()/scale << endl;
  
  
  output_header << "!END OF INTERFILE :=\n";
  
  
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
    // KT 05/11/98 added ! 
    output_header << "!imagedata byte order := " <<
      (ByteOrder::get_native_order() == ByteOrder::little_endian 
      ? "LITTLEENDIAN"
      : "BIGENDIAN")
      << endl;

    // KT 09/11/98 use output_type
    output_header << "!number format := ";
    if (output_type.integer_type() )
      output_header << (output_type.signed_type() 
      ? "signed integer\n" : "unsigned integer\n");
    else
      output_header << "float\n";
    output_header << "!number of bytes per pixel := "
      << output_type.size_in_bytes() << endl;
    
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

// KT 09/11/98 new
bool
write_basic_interfile(const char * const filename, 
		      const PETImageOfVolume& image,
		      const NumericType output_type)
{
  return
    write_basic_interfile(filename, 
                          image, // use automatic reference to base class
			  image.get_voxel_size(), 
			  output_type);
}

PETSinogramOfVolume read_interfile_PSOV(istream& input)
{
  //KT 26/10/98 new
  // Horrible trick to keep the stream alive. At the moment, this pointer
  // is never deleted
  // TODO
  fstream *data_in =  new fstream;
  

  // KT 13/11/98 moved stream arg from constructor to parse()
  InterfilePSOVHeader hdr;
  if (!hdr.parse(input))
    {
      PETerror("\nError parsing interfile header, \n\
     I am going to ask you lots of questions...\n");
      return ask_PSOV_details(data_in);
    }
   
    open_read_binary(*data_in, hdr.data_file_name.c_str());

    vector<int> min_ring_num_per_segment(hdr.num_segments);
    vector<int> max_ring_num_per_segment(hdr.num_segments);
    for (int s=0; s<hdr.num_segments; s++)
    {
      min_ring_num_per_segment[s] = 0;
      max_ring_num_per_segment[s] = hdr.num_rings_per_segment[s]-1;
    }

    // TODO Horrible trick to get the scanner right
    PETScannerInfo scanner;
    switch (hdr.num_views)
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
    // TODO scaling factors
    for (int i=1; i<hdr.image_scaling_factors[0].size(); i++)
      if (hdr.image_scaling_factors[0][0] != hdr.image_scaling_factors[0][i])
      { 
        PETerror("Interfile warning: all image scaling factors should be equal \n\
at the moment. Using the first scale factor only.\n");
        break;
      }
    return PETSinogramOfVolume(
                      scanner,
		      hdr.segment_sequence,
		      hdr.min_ring_difference, 
		      hdr.max_ring_difference, 
		      min_ring_num_per_segment,
		      max_ring_num_per_segment,
		      0, hdr.num_views-1,
		      -hdr.num_bins/2,
		      (-hdr.num_bins/2) + hdr.num_bins-1,
		      *data_in,
		      hdr.data_offset[0],
		      hdr.storage_order,
		      hdr.type_of_numbers,
		      hdr.image_scaling_factors[0][0]);
}



// KT 13/11/98 new
PETSinogramOfVolume read_interfile_PSOV(const char *const filename)
{
    ifstream image_stream(filename);
    if (!image_stream)
    { 
      PETerror("read_interfile_PSOV: couldn't open file %s\n", filename);
      exit(1);
    }

  return read_interfile_PSOV(image_stream);
}


/**********************************************************************
 template instantiations
 **********************************************************************/

// KT 01/11/98 disabled 2 instantiations
/*
template bool write_basic_interfile<>(const char * const filename, 
				      const Tensor3D<signed char>&,
				      const Point3D& voxel_size,
				      const NumericType output_type );
template bool write_basic_interfile<>(const char * const filename, 
				      const Tensor3D<unsigned char>&,const Point3D& voxel_size,
				      const NumericType output_type);
*/
template bool write_basic_interfile<>(const char * const filename, 
				      const Tensor3D<signed short>&,
				      const Point3D& voxel_size,
				      const NumericType output_type);
template bool write_basic_interfile<>(const char * const filename, 
				      const Tensor3D<unsigned short>&,
				      const Point3D& voxel_size,
				      const NumericType output_type);
template bool write_basic_interfile<>(const char * const filename, 
				      const Tensor3D<float>&,
				      const Point3D& voxel_size,
				      const NumericType output_type);
