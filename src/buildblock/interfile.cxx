//
// $Id$ :$Date$
//

/* 
  Functions which read/write Interfile images
  Pretty horrible implementations at the moment...

  First version: Kris Thielemans
   */

#include "InterfileHeader.h"
#include "interfile.h"
#include "utilities.h"

// KT 14/10/98 make arg istream
// KT 14/01/2000 added directory capability
PETImageOfVolume read_interfile_image(istream& input, 
				      const char * const directory_for_data)
{
  InterfileImageHeader hdr;
  if (!hdr.parse(input))
    {
      PETerror("\nError parsing interfile header, \n\
               I am going to ask you lots of questions...\n");
      return ask_image_details();
    }
  
  // KT 14/01/2000 added directory capability
  // prepend directory_for_data to the data_file_name from the header
  char full_data_file_name[max_filename_length];
  strcpy(full_data_file_name, hdr.data_file_name.c_str());
  prepend_directory_name(full_data_file_name, directory_for_data);  
  
  ifstream data_in;
  open_read_binary(data_in, full_data_file_name);
  
  Point3D origin(0,0,0);
  
  Point3D voxel_size(hdr.pixel_sizes[0], hdr.pixel_sizes[1], hdr.pixel_sizes[2]);
  
  // KT 29/10/98 adjusted x,y ranges to centre 
  // KT 03/11/99 added int
  const int y_size =  hdr.matrix_size[1][0];
  const int x_size =  hdr.matrix_size[0][0];
  PETImageOfVolume 
    image(Tensor3D<float>(0,hdr.matrix_size[2][0]-1,
                          -y_size/2, (-y_size/2) + y_size - 1,
                          -x_size/2, (-x_size/2) + x_size - 1), 
          origin,
          voxel_size);
  
  data_in.seekg(hdr.data_offset[0]);
  
  Real scale = Real(1);
  image.read_data(data_in, hdr.type_of_numbers, scale, hdr.file_byte_order);
  // TODO perform run-time check also when not debugging
  assert(scale == 1);  
  
  for (int i=0; i< hdr.matrix_size[2][0]; i++)
    if (hdr.image_scaling_factors[0][i]!= 1)
      image[i] *= hdr.image_scaling_factors[0][i];
    
  return image;
}


PETImageOfVolume read_interfile_image(const char *const filename)
{
  ifstream image_stream(filename);
  if (!image_stream)
  { 
    PETerror("read_interfile_image: couldn't open file %s\n", filename);
    exit(1);
  }

  // KT 14/01/2000 added directory capability
  char directory_name[max_filename_length];
  get_directory_name(directory_name, filename);
  
  return read_interfile_image(image_stream, directory_name);
}


const VectorWithOffset<unsigned long> 
compute_file_offsets(int number_of_time_frames,
	     const NumericType output_type,
	     const Coordinate3D<int>& dim,
	     unsigned long initial_offset)
{ 
  const unsigned long a= dim.x*dim.y*dim.z*output_type.size_in_bytes();
  VectorWithOffset<unsigned long> temp(number_of_time_frames);
  {
  for (int i=0; i<=number_of_time_frames-1;i++)
      temp[i]= initial_offset +i*a;
  }
  return temp;
}
 
bool 
write_basic_interfile_image_header(const string& header_file_name,
				   const string& image_file_name,
				   const Coordinate3D<int>& dimensions, 
				   const Point3D& voxel_size,
			           const NumericType output_type,
				   const ByteOrder byte_order,
				   const VectorWithOffset<float>& scaling_factors,
				   const VectorWithOffset<unsigned long>& file_offsets)
{ 
 
  string header_name = header_file_name;
  ofstream output_header(header_name.c_str(), ios::out);
  if (!output_header.good())
  {
    cerr << "Error opening Interfile header '" 
         << header_name << " for writing" << endl;
    return false;
  }  

  output_header << "!INTERFILE  :=\n";
  output_header << "name of data file := " << image_file_name << endl;
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
  if (output_type.integer_type() )
    output_header << (output_type.signed_type() 
                      ? "signed integer\n" : "unsigned integer\n");
  else
    output_header << "float\n";
  output_header << "!number of bytes per pixel := "
       << output_type.size_in_bytes() << endl;
  
  output_header << "number of dimensions := 3\n";

  output_header << "!matrix size [1] := "
                << dimensions.x << endl;
  output_header << "matrix axis label [1] := x\n";
  output_header << "scaling factor (mm/pixel) [1] := " 
                << voxel_size.x << endl;
  output_header << "!matrix size [2] := "
                << dimensions.y<< endl;
  output_header << "matrix axis label [2] := y\n";
  output_header << "scaling factor (mm/pixel) [2] := " 
                << voxel_size.y << endl;
  output_header << "!matrix size [3] := "
                << dimensions.z<< endl;
  output_header << "matrix axis label [3] := z\n";
  output_header << "scaling factor (mm/pixel) [3] := " 
                << voxel_size.z << endl;

  output_header << "number of time frames := " << scaling_factors.get_length() << endl;
  
  for (int i=1; i<=scaling_factors.get_length();i++)
  {
  output_header << "image scaling factor"<<"["<<i<<"] := " << scaling_factors[i-1]<< endl;
    // KT&SM 14/01/2000 added    
  output_header << "data offset in bytes"<<"["<<i<<"] := "<< file_offsets[i-1]<< endl;
  }

  // analogue of image scaling factor[*] for Louvain la Neuve Interfile reader
  if (scaling_factors.get_length() > 1)
  {
    const float first_scaling_factor = scaling_factors[0];
    for (int i=1; i <= scaling_factors.get_max_index(); i++)
    {
      if (scaling_factors[i] != first_scaling_factor)
      {
	cerr << "Interfile WARNING: quantification units keyword is set incorrectly\n";
	break;
      }
    }
  }
  output_header << "quantification units := " << scaling_factors[0] << endl;
  // TODO
  // output_header << "maximum pixel count := " << image.find_max()/scale << endl;  
  output_header << "!END OF INTERFILE :=\n";

  
    // temporary copy to make an old-style header to satisfy Analyze
  {
    string header_name = header_file_name;
    // TODO replace old extension with new one
    header_name += ".ahv";
    ofstream output_header(header_name.c_str(), ios::out);
    if (!output_header.good())
    {
      cerr << "Error opening old-style Interfile header '" 
	<< header_name << " for writing" << endl;
      return false;
    }  
    
    output_header << "!INTERFILE  :=\n";
    output_header << "!name of data file := " << image_file_name << endl;
    output_header << "!total number of images := "
                  << dimensions.z << endl;
    // KT&SM 14/01/2000 added
    for (int i=1;i<=file_offsets.get_length();i++)
    {
      output_header << "!data offset in bytes := " <<file_offsets[i-1]<< endl;
    }
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
      << dimensions.x<< endl;
    output_header << "matrix axis label [1] := x\n";
    // KT 16/02/98 added voxel size
    output_header << "scaling factor (mm/pixel) [1] := " 
                << voxel_size.x << endl;
    output_header << "!matrix size [2] := "
      << dimensions.y << endl;
    output_header << "matrix axis label [2] := y\n";
    // KT 16/02/98 added voxel size
    output_header << "scaling factor (mm/pixel) [2] := " 
                << voxel_size.y << endl;
    // KT 16/02/98 added voxel size
    {
      // Note: bug in current version of analyze
      // if voxel_size is not an integer, it will not take the 
      // pixel size into account
      // Work around: Always make sure it is not an integer, by
      // adding a small number to it if necessary
      // TODO this is horrible and not according to the Interfile standard
      // so, remove for distribution purposes
      float zsize = voxel_size.z;
      if (floor(zsize)==zsize)
	zsize += 0.00001F;
      // TODO this is what it should be
      // float zsize = voxel_size.z/ voxel_size.x;
      
      output_header << "!slice thickness (pixels) := " 
                << zsize << endl;
    }
    output_header << "!END OF INTERFILE :=\n";
    
  }
  return true;
}


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
    data_name += ".v";
    
    output_header << "!INTERFILE  :=\n";
    output_header << "!name of data file := " << data_name << endl;
    output_header << "!total number of images := "
      << image.get_length3() << endl;
    output_header << "!imagedata byte order := " <<
      (ByteOrder::get_native_order() == ByteOrder::little_endian 
      ? "LITTLEENDIAN"
      : "BIGENDIAN")
      << endl;

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
    output_header << "scaling factor (mm/pixel) [1] := " 
                << voxel_size.x << endl;
    output_header << "!matrix size [2] := "
      << image.get_length2() << endl;
    output_header << "matrix axis label [2] := y\n";
    output_header << "scaling factor (mm/pixel) [2] := " 
                << voxel_size.y << endl;
    {
      // Note: bug in current version of analyze
      // if voxel_size is not an integer, it will not take the 
      // pixel size into account
      // Work around: Always make sure it is not an integer, by
      // adding a small number to it if necessary
      // TODO this is horrible and not according to the Interfile standard
      // so, remove for distribution purposes
      float zsize = voxel_size.z;
      if (floor(zsize)==zsize)
	zsize += 0.00001F;
      // TODO this is what it should be
      // float zsize = voxel_size.z/ voxel_size.x;
      
      output_header << "!slice thickness (pixels) := " 
                << zsize << endl;
    }
    output_header << "!END OF INTERFILE :=\n";
    
  }
  return true;
}

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

// KT 14/01/2000 added directory capability
PETSinogramOfVolume 
read_interfile_PSOV(istream& input,
		    const char * const directory_for_data)
{
  // Horrible trick to keep the stream alive. At the moment, this pointer
  // is never deleted
  // TODO
  fstream *data_in =  new fstream;
  

  InterfilePSOVHeader hdr;
  if (!hdr.parse(input))
  {
      PETerror("\nError parsing interfile header, \n\
     I am going to ask you lots of questions...\n");
      return ask_PSOV_details(data_in);
  }

  // KT 14/01/2000 added directory capability
  // prepend directory_for_data to the data_file_name from the header
  char full_data_file_name[max_filename_length];
  strcpy(full_data_file_name, hdr.data_file_name.c_str());
  prepend_directory_name(full_data_file_name, directory_for_data);  
  
  open_read_binary(*data_in, full_data_file_name);
  
  vector<int> min_ring_num_per_segment(hdr.num_segments);
  vector<int> max_ring_num_per_segment(hdr.num_segments);
  for (int s=0; s<hdr.num_segments; s++)
  {
    min_ring_num_per_segment[s] = 0;
    max_ring_num_per_segment[s] = hdr.num_rings_per_segment[s]-1;
  }
  
  for (int i=1; i<hdr.image_scaling_factors[0].size(); i++)
    if (hdr.image_scaling_factors[0][0] != hdr.image_scaling_factors[0][i])
    { 
      PETerror("Interfile warning: all image scaling factors should be equal \n\
at the moment. Using the first scale factor only.\n");
      break;
    }

  return PETSinogramOfVolume(
                      hdr.scan_info,
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
		      hdr.file_byte_order,
		      hdr.image_scaling_factors[0][0]);
}



PETSinogramOfVolume 
read_interfile_PSOV(const char *const filename)
{
  ifstream image_stream(filename);
  if (!image_stream)
  { 
    PETerror("read_interfile_PSOV: couldn't open file %s\n", filename);
    exit(1);
  }
  
  // KT 14/01/2000 added directory capability
  char directory_name[max_filename_length];
  get_directory_name(directory_name, filename);
  
  return read_interfile_PSOV(image_stream, directory_name);
}

// TODO add directory capability
bool 
write_basic_interfile_PSOV_header(const string& header_file_name,
				  const string& image_file_name,
				  const PETSinogramOfVolume& psov)
{

  ofstream output_header(header_file_name.c_str(), ios::out);
  if (!output_header.good())
  {
    cerr << "Error opening Interfile header '" 
         << header_file_name << " for writing" << endl;
    return false;
  }  


  output_header << "!INTERFILE  :=\n";
  output_header << "name of data file := " <<image_file_name << endl;

  output_header << "originating system := ";
  switch(psov.scan_info.get_scanner().type)
    {
      case PETScannerInfo::RPT: output_header << "PRT-1\n"; break;
      case PETScannerInfo::E931: output_header << "ECAT 931\n"; break;
      case PETScannerInfo::E951: output_header << "ECAT 951\n"; break;

      case PETScannerInfo::E953: output_header << "ECAT 953\n"; break;
      case PETScannerInfo::E966: output_header << "ECAT 966\n"; break;
      case PETScannerInfo::ART: output_header << "ECAT ART\n"; break;
      case PETScannerInfo::Advance: output_header << "Advance\n"; break;
      default: output_header << "Unknown\n"; break;
    }

  output_header << "!GENERAL DATA :=\n";
  output_header << "!GENERAL IMAGE DATA :=\n";
  output_header << "!type of data := PET\n";
  // TODO use on_disk_byte_order
  output_header << "imagedata byte order := " <<
    (ByteOrder::get_native_order() == ByteOrder::little_endian 
     ? "LITTLEENDIAN"
     : "BIGENDIAN")
    << endl;

  output_header << "!PET STUDY (General) :=\n";
  output_header << "!PET data type := Emission\n";
  // TODO hard-wired float
  output_header << "!number format := float\n";
  output_header << "!number of bytes per pixel := 4\n";

  output_header << "number of dimensions := 4\n";
  
  // TODO support more ? 
  {
    // default to SegmentViewRingBin
    int order_of_segment = 1;
    int order_of_view = 2;
    int order_of_z = 3;
    int order_of_bin = 4;
    switch(psov.get_storage_order())
    {  
    case PETSinogramOfVolume::ViewSegmentRingBin:
      {
	order_of_segment = 2;
	order_of_view = 1;
	order_of_z = 3;
	break;
      }
    case PETSinogramOfVolume::SegmentViewRingBin:
      {
	order_of_segment = 1;
	order_of_view = 2;
	order_of_z = 3;
	break;
      }
    case PETSinogramOfVolume::SegmentRingViewBin:
      {
	order_of_segment = 1;
	order_of_view = 3;
	order_of_z = 2;
	break;
      }
    default:
      {
      PETerror("write_interfile_PSOV_header: unsupported storage order,\
defaulting to SegmentViewRingBin.\n Please correct by hand !");
      }
    }
    
    output_header << "!matrix size [" << order_of_segment << "] := " 
      << psov.get_num_segments() << "\n";
    output_header << "matrix axis label [" << order_of_segment 
      << "] := segment\n";
    output_header << "!matrix size [" << order_of_view << "] := "
      << psov.get_num_views() << "\n";
    output_header << "matrix axis label [" << order_of_view << "] := view\n";
    output_header << "!matrix size [" << order_of_z << "] := "
      << "{" << psov.get_num_rings(0);
    for (int s=1; s<= psov.get_max_segment(); s++)
      output_header << "," << psov.get_num_rings(s) << "," << psov.get_num_rings(-s);
    output_header << "}\n";
    
    output_header << "matrix axis label [" << order_of_z << "] := z\n";
    output_header << "!matrix size [" << order_of_bin << "] := "
      << psov.get_num_bins() << "\n";
    output_header << "matrix axis label [" << order_of_bin << "] := bin\n";
  }

  output_header << "minimum ring difference per segment := {"
    << psov.get_min_ring_difference(0);
  for (int s=1; s<= psov.get_max_segment(); s++)
    output_header << "," << psov.get_min_ring_difference(s) 
                  << "," << psov.get_min_ring_difference(-s);
  output_header << "}\n";
  output_header << "maximum ring difference per segment := {0";
  for (int s=1; s<= psov.get_max_segment(); s++)
    output_header << "," << psov.get_max_ring_difference(s) 
                  << "," << psov.get_max_ring_difference(-s);
  output_header << "}\n";
  
  output_header << "number of rings := " 
    << psov.scan_info.get_num_rings() << endl;
  output_header << "number of detectors per ring := " 
    << psov.scan_info.get_num_views()*2 << endl;
  output_header << "transaxial FOV diameter (cm) := "
    << psov.scan_info.get_ring_radius()*2/10. << endl;
  output_header << "distance between rings (cm) := " 
    << psov.scan_info.get_ring_spacing()/10. << endl;
  output_header << "bin size (cm) := " 
    << psov.scan_info.get_bin_size()/10. << endl;
  output_header << "view offset (degrees) := "
    << psov.scan_info.get_view_offset() << endl;

  output_header << "number of time frames := 1\n";
  output_header << "!END OF INTERFILE :=\n";

  return true;
}

/**********************************************************************
 template instantiations
 **********************************************************************/

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
