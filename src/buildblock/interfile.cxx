//
// $Id$: $Date$
//

/*!
  \file 
  \ingroup buildblock
 
  \brief  Implementation of functions which read/write Interfile data

  \author Kris Thielemans 
  \author Sanida Mustafovic
  \author PARAPET project

  \date    $Date$
  \version $Revision$
    
*/
//   Pretty horrible implementations at the moment...

#include "interfile.h"
#include "InterfileHeader.h"
#include "IndexRange3D.h"
#include "utilities.h"
#include "CartesianCoordinate3D.h"
#include "VoxelsOnCartesianGrid.h"
#include "ProjDataFromStream.h"
#include "ProjDataInfoCylindrical.h"
#include "Scanner.h"

#include <fstream>

#ifndef TOMO_NO_NAMESPACES
using std::cerr;
using std::endl;
using std::ofstream;
using std::ifstream;
using std::fstream;
#endif

START_NAMESPACE_TOMO


VoxelsOnCartesianGrid<float>* read_interfile_image(istream& input, 
				      const string&  directory_for_data)
{
  InterfileImageHeader hdr;
  if (!hdr.parse(input))
  {
      warning("\nError parsing interfile header, \n\
I am going to ask you lots of questions...\n");
      return new VoxelsOnCartesianGrid<float>
        (VoxelsOnCartesianGrid<float>::ask_parameters());
    }
  
  // prepend directory_for_data to the data_file_name from the header
  char full_data_file_name[max_filename_length];
  strcpy(full_data_file_name, hdr.data_file_name.c_str());
  prepend_directory_name(full_data_file_name, directory_for_data.c_str());  
  
  ifstream data_in;
  open_read_binary(data_in, full_data_file_name);
  
  CartesianCoordinate3D<float> origin(0,0,0);
  
  CartesianCoordinate3D<float> voxel_size(hdr.pixel_sizes[2], hdr.pixel_sizes[1], hdr.pixel_sizes[0]);
  
  // KT 29/10/98 adjusted x,y ranges to centre 
  // KT 03/11/99 added int
  const int y_size =  hdr.matrix_size[1][0];
  const int x_size =  hdr.matrix_size[0][0];
  VoxelsOnCartesianGrid<float>* image_ptr =
    new VoxelsOnCartesianGrid<float>
            (IndexRange3D(0,hdr.matrix_size[2][0]-1,
			  -y_size/2, (-y_size/2) + y_size - 1,
			  -x_size/2, (-x_size/2) + x_size - 1), 
	  origin,
	  voxel_size);
  
  data_in.seekg(hdr.data_offset[0]);
  
  float scale = float(1);
  image_ptr->read_data(data_in, hdr.type_of_numbers, scale, hdr.file_byte_order);
  // TODO perform run-time check also when not debugging
  assert(scale == 1);  
  
  for (int i=0; i< hdr.matrix_size[2][0]; i++)
    if (hdr.image_scaling_factors[0][i]!= 1)
      (*image_ptr)[i] *= hdr.image_scaling_factors[0][i];
    
  return image_ptr;
}


VoxelsOnCartesianGrid<float>* read_interfile_image(const string& filename)
{
  ifstream image_stream(filename.c_str());
  if (!image_stream)
    { 
      error("read_interfile_image: couldn't open file %s\n", filename.c_str());
    }

  char directory_name[max_filename_length];
  get_directory_name(directory_name, filename.c_str());
  
  return read_interfile_image(image_stream, directory_name);
}


const VectorWithOffset<unsigned long> 
compute_file_offsets(int number_of_time_frames,
		     const NumericType output_type,
		     const CartesianCoordinate3D<int>& dim,
		     unsigned long initial_offset)
{ 
  const unsigned long a= dim.x()*dim.y()*dim.z()*output_type.size_in_bytes();
  VectorWithOffset<unsigned long> temp(number_of_time_frames);
  {
    for (int i=0; i<=number_of_time_frames-1;i++)
      temp[i]= initial_offset +i*a;
  }
  return temp;
}
 
Succeeded 
write_basic_interfile_image_header(const string& header_file_name,
				   const string& image_file_name,
				   const CartesianCoordinate3D<int>& dimensions, 
				   const CartesianCoordinate3D<float>& voxel_size,
				   const NumericType output_type,
				   const ByteOrder byte_order,
				   const VectorWithOffset<float>& scaling_factors,
				   const VectorWithOffset<unsigned long>& file_offsets)
{ 
 
  char *header_name = new char[header_file_name.size() +5];
  strcpy(header_name, header_file_name.c_str());
  add_extension(header_name, ".hv");
  ofstream output_header(header_name, ios::out);
  if (!output_header.good())
    {
      cerr << "Error opening Interfile header '" 
	   << header_name << " for writing" << endl;
      return Succeeded::no;
    }  
  delete header_name;

  output_header << "!INTERFILE  :=\n";
  output_header << "name of data file := " << image_file_name << endl;
  output_header << "!GENERAL DATA :=\n";
  output_header << "!GENERAL IMAGE DATA :=\n";
  output_header << "!type of data := PET\n";
  output_header << "imagedata byte order := " <<
    (byte_order == ByteOrder::little_endian 
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

  output_header << "matrix axis label [1] := x\n";
  output_header << "!matrix size [1] := "
		<< dimensions.x() << endl;
  output_header << "scaling factor (mm/pixel) [1] := " 
		<< voxel_size.x() << endl;
  output_header << "matrix axis label [2] := y\n";
  output_header << "!matrix size [2] := "
		<< dimensions.y() << endl;
  output_header << "scaling factor (mm/pixel) [2] := " 
		<< voxel_size.y() << endl;
  output_header << "matrix axis label [3] := z\n";
  output_header << "!matrix size [3] := "
		<< dimensions.z()<< endl;
  output_header << "scaling factor (mm/pixel) [3] := " 
		<< voxel_size.z() << endl;

  output_header << "number of time frames := " << scaling_factors.get_length() << endl;
  
  for (int i=1; i<=scaling_factors.get_length();i++)
    {
      output_header << "image scaling factor"<<"["<<i<<"] := " << scaling_factors[i-1]<< endl;
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
    char *header_name = new char[header_file_name.size() +5];
    strcpy(header_name, header_file_name.c_str());
    replace_extension(header_name, ".ahv");

    ofstream output_header(header_name, ios::out);
    if (!output_header.good())
      {
	cerr << "Error opening old-style Interfile header '" 
	     << header_name << " for writing" << endl;
        return Succeeded::no;
      }  
    delete header_name;
    
    output_header << "!INTERFILE  :=\n";
    output_header << "!name of data file := " << image_file_name << endl;
    output_header << "!total number of images := "
		  << dimensions.z() << endl;
    for (int i=1;i<=file_offsets.get_length();i++)
      {
	output_header << "!data offset in bytes := " <<file_offsets[i-1]<< endl;
      }
    output_header << "!imagedata byte order := " <<
      (byte_order == ByteOrder::little_endian 
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
    
    output_header << "matrix axis label [1] := x\n";
    output_header << "!matrix size [1] := "
		  << dimensions.x()<< endl;
    output_header << "scaling factor (mm/pixel) [1] := " 
		  << voxel_size.x() << endl;
    output_header << "matrix axis label [2] := y\n";
    output_header << "!matrix size [2] := "
		  << dimensions.y() << endl;
    output_header << "scaling factor (mm/pixel) [2] := " 
		  << voxel_size.y() << endl;
    {
      // Note: bug in current version of analyze
      // if voxel_size is not an integer, it will not take the 
      // pixel size into account
      // Work around: Always make sure it is not an integer, by
      // adding a small number to it if necessary
      // TODO this is horrible and not according to the Interfile standard
      // so, remove for distribution purposes
      float zsize = voxel_size.z();
      if (floor(zsize)==zsize)
	zsize += 0.00001F;
      // TODO this is what it should be
      // float zsize = voxel_size.z()/ voxel_size.x();
      
      output_header << "!slice thickness (pixels) := " 
		    << zsize << endl;
    }
    output_header << "!END OF INTERFILE :=\n";
    
  }
  return Succeeded::yes;
}



template <class elemT>
Succeeded 
write_basic_interfile(const string& filename, 
		      const Array<3,elemT>& image,
		      const NumericType output_type)
{
  return
    write_basic_interfile(filename, 
			  image, 
			  CartesianCoordinate3D<float>(1,1,1), 
			  output_type);
}

template <class NUMBER>
Succeeded write_basic_interfile(const string&  filename, 
			   const Array<3,NUMBER>& image,
			   const CartesianCoordinate3D<float>& voxel_size,
			   const NumericType output_type)
{
  CartesianCoordinate3D<int> min_indices;
  CartesianCoordinate3D<int> max_indices;
  if (!image.get_regular_range(min_indices, max_indices))
  {
    warning("write_basic_interfile: can handle only regular index ranges\n. No output\n");
    return Succeeded::no;
  }
  CartesianCoordinate3D<int> dimensions = max_indices - min_indices;
  dimensions += 1;

  char * data_name = new char[filename.size() + 5];
  char * header_name = new char[filename.size() + 5];

  strcpy(data_name, filename.c_str());
  add_extension(data_name, ".v");
  strcpy(header_name, data_name);
  replace_extension(header_name, ".hv");

  ofstream output_data;
  open_write_binary(output_data, data_name);

  float scale = 0;
  image.write_data(output_data, output_type, scale);

  VectorWithOffset<float> scaling_factors(1);
  scaling_factors[0] = scale;
  VectorWithOffset<unsigned long> file_offsets(1);
  file_offsets.fill(0);


  write_basic_interfile_image_header(header_name,
				   data_name,
				   dimensions, 
				   voxel_size,
				   output_type,
				   ByteOrder::native,
				   scaling_factors,
				   file_offsets);
  delete header_name;
  delete data_name;

  return Succeeded::yes;
}

Succeeded
write_basic_interfile(const string&  filename, 
		      const VoxelsOnCartesianGrid<float>& image,
		      const NumericType output_type)
{
  return
    write_basic_interfile(filename, 
			  image, // use automatic reference to base class
			  image.get_grid_spacing(), 
			  output_type);
}

Succeeded 
write_basic_interfile(const string& filename, 
		      const DiscretisedDensity<3,float>& image,
		      const NumericType output_type)
{
  // dynamic_cast will throw an exception when it's not valid
  return
    write_basic_interfile(filename,
                          dynamic_cast<const VoxelsOnCartesianGrid<float>& >(image),
                          output_type);
}



ProjDataFromStream* 
read_interfile_PDFS(istream& input,
		    const string& directory_for_data,
		    const ios::openmode open_mode)
{
  
  InterfilePDFSHeader hdr;  

   if (!hdr.parse(input))
    {
      warning("\nError parsing interfile header, \n\
I am going to ask you lots of questions...\n");
      return ProjDataFromStream::ask_parameters();
    }

  // KT 14/01/2000 added directory capability
  // prepend directory_for_data to the data_file_name from the header

  char full_data_file_name[max_filename_length];
  strcpy(full_data_file_name, hdr.data_file_name.c_str());
  prepend_directory_name(full_data_file_name, directory_for_data.c_str());  
 
  vector<int> min_ring_num_per_segment(hdr.num_segments);
  vector<int> max_ring_num_per_segment(hdr.num_segments);
  for (int s=0; s<hdr.num_segments; s++)
    {
      min_ring_num_per_segment[s] = 0;
      max_ring_num_per_segment[s] = hdr.num_rings_per_segment[s]-1;
    }
  
  for (unsigned int i=1; i<hdr.image_scaling_factors[0].size(); i++)
    if (hdr.image_scaling_factors[0][0] != hdr.image_scaling_factors[0][i])
      { 
	warning("Interfile warning: all image scaling factors should be equal \n\
at the moment. Using the first scale factor only.\n");
	break;
      }
  
   assert(hdr.data_info_ptr !=0);

   fstream *data_in = 
     new fstream (full_data_file_name, open_mode | ios::binary);
   if (!data_in->good())
     {
       error("error opening file %s\n",full_data_file_name);
     }

  return new ProjDataFromStream(hdr.data_info_ptr,			    
			        data_in,
			        hdr.data_offset[0],
                                hdr.segment_sequence,
			        hdr.storage_order,
			        hdr.type_of_numbers,
			        hdr.file_byte_order,
			        hdr.image_scaling_factors[0][0]);


}


ProjDataFromStream*
read_interfile_PDFS(const string& filename,
		    const ios::openmode open_mode)
{
  ifstream image_stream(filename.c_str());
  if (!image_stream)
    { 
      error("read_interfile_PDFS: couldn't open file %s\n", filename.c_str());
    }
  
  char directory_name[max_filename_length];
  get_directory_name(directory_name, filename.c_str());
  
  return read_interfile_PDFS(image_stream, directory_name, open_mode);
}

// TODO add directory capability
Succeeded 
write_basic_interfile_PDFS_header(const string& header_file_name,
				  const string& data_file_name,
				  const ProjDataFromStream& pdfs)
{

  char *header_name = new char[header_file_name.size() +5];
  strcpy(header_name, header_file_name.c_str());
  add_extension(header_name, ".hs");
  ofstream output_header(header_name, ios::out);
  if (!output_header.good())
    {
      cerr << "Error opening Interfile header '" 
	   << header_name << " for writing" << endl;
      return Succeeded::no;
    }  
  delete header_name;

  const vector<int> segment_sequence = pdfs.get_segment_sequence_in_stream();

  output_header << "!INTERFILE  :=\n";
  output_header << "name of data file := " <<data_file_name << endl;

  output_header << "originating system := ";
  output_header <<pdfs.get_proj_data_info_ptr()->get_scanner_ptr()->get_name() << endl;

  output_header << "!GENERAL DATA :=\n";
  output_header << "!GENERAL IMAGE DATA :=\n";
  output_header << "!type of data := PET\n";
  output_header << "imagedata byte order := " <<
    (pdfs.get_byte_order_in_stream() == ByteOrder::little_endian 
     ? "LITTLEENDIAN"
     : "BIGENDIAN")
		<< endl;

  output_header << "!PET STUDY (General) :=\n";
  output_header << "!PET data type := Emission\n";
  {
    string number_format;
    size_t size_in_bytes;

    const NumericType data_type = pdfs.get_data_type_in_stream();
    data_type.get_Interfile_info(number_format, size_in_bytes);

    output_header << "!number format := " << number_format << "\n";
    output_header << "!number of bytes per pixel := " << size_in_bytes << "\n";
  }

  output_header << "number of dimensions := 4\n";
  
  // TODO support more ? 
  {
    // default to Segment_View_AxialPos_TangPos
    int order_of_segment = 4;
    int order_of_view = 3;
    int order_of_z = 2;
    int order_of_bin = 1;
    switch(pdfs.get_storage_order())
     /*
      {  
      case ProjDataFromStream::ViewSegmentRingBin:
	{
	  order_of_segment = 2;
	  order_of_view = 1;
	  order_of_z = 3;
	  break;
	}
	*/
    {
      case ProjDataFromStream::Segment_View_AxialPos_TangPos:
	{
	  order_of_segment = 4;
	  order_of_view = 3;
	  order_of_z = 2;
	  break;
	}
      case ProjDataFromStream::Segment_AxialPos_View_TangPos:
	{
	  order_of_segment = 4;
	  order_of_view = 2;
	  order_of_z = 3;
	  break;
	}
      default:
	{
	  error("write_interfile_PSOV_header: unsupported storage order,\
defaulting to Segment_View_AxialPos_TangPos.\n Please correct by hand !");
	}
      }
    
    output_header << "matrix axis label [" << order_of_segment 
		  << "] := segment\n";
    output_header << "!matrix size [" << order_of_segment << "] := " 
		  << pdfs.get_segment_sequence_in_stream().size()<< "\n";
    output_header << "matrix axis label [" << order_of_view << "] := view\n";
    output_header << "!matrix size [" << order_of_view << "] := "
		  << pdfs.get_proj_data_info_ptr()->get_num_views() << "\n";
    
    output_header << "matrix axis label [" << order_of_z << "] := axial coordinate\n";
    output_header << "!matrix size [" << order_of_z << "] := ";
    // tedious way to print a list of numbers
    {
#ifndef TOMO_NO_NAMESPACES
      // VC needs this explicitly here
      std::
#endif
      vector<int>::const_iterator seg = segment_sequence.begin();
      output_header << "{ " <<pdfs.get_proj_data_info_ptr()->get_num_axial_poss(*seg);
      for (seg++; seg != segment_sequence.end(); seg++)
	output_header << "," << pdfs.get_proj_data_info_ptr()->get_num_axial_poss(*seg);
      output_header << "}\n";
    }

    output_header << "matrix axis label [" << order_of_bin << "] := tangential coordinate\n";
    output_header << "!matrix size [" << order_of_bin << "] := "
		  <<pdfs.get_proj_data_info_ptr()->get_num_tangential_poss() << "\n";
  }

  const  ProjDataInfoCylindrical* proj_data_info_ptr = 
    dynamic_cast< const  ProjDataInfoCylindrical*> (pdfs.get_proj_data_info_ptr());

   if (proj_data_info_ptr!=NULL)
     {
       // cylindrical scanners
   
       output_header << "minimum ring difference per segment := ";    
       {
#ifndef TOMO_NO_NAMESPACES
	 // VC needs this explicitly here
	 std::
#endif
	   vector<int>::const_iterator seg = segment_sequence.begin();
	 output_header << "{ " << proj_data_info_ptr->get_min_ring_difference(*seg);
	 for (seg++; seg != segment_sequence.end(); seg++)
	   output_header << "," <<proj_data_info_ptr->get_min_ring_difference(*seg);
	 output_header << "}\n";
       }

       output_header << "maximum ring difference per segment := ";
       {
#ifndef TOMO_NO_NAMESPACES
	 // VC needs this explicitly here
	 std::
#endif
	   vector<int>::const_iterator seg = segment_sequence.begin();
	 output_header << "{ " <<proj_data_info_ptr->get_max_ring_difference(*seg);
	 for (seg++; seg != segment_sequence.end(); seg++)
	   output_header << "," <<proj_data_info_ptr->get_max_ring_difference(*seg);
	 output_header << "}\n";
       }
  
       output_header << "number of rings := " 
		     << pdfs.get_proj_data_info_ptr()->get_scanner_ptr()->get_num_rings() << endl;
       output_header << "number of detectors per ring := " 
		     << pdfs.get_proj_data_info_ptr()->get_scanner_ptr()->get_max_num_views()*2 << endl;

       // KT 18/01/2001 use data from proj_data_info instead of scanner
       output_header << "ring diameter (cm) := "
		     << proj_data_info_ptr->get_ring_radius()*2/10. << endl;
       output_header << "distance between rings (cm) := " 
		     << proj_data_info_ptr->get_ring_spacing()/10. << endl;
       output_header << "bin size (cm) := " 
		     << proj_data_info_ptr->get_sampling_in_s(Bin(0,0,0,0))/10. << endl;
		     //pdfs.get_proj_data_info_ptr()->get_scanner_ptr()->get_default_bin_size()/10. << endl;

       //output_header << "view offset (degrees) := "
       //		<< pdfs.get_proj_data_info_ptr()->get_scanner_ptr()->view_offset<< endl;

     } // end of cylindrical scanner
  else
    {
      // TODO something here
    }

  output_header<<"data offset in bytes[1] := "
	       <<pdfs.get_offset_in_stream()<<endl;
    

  output_header << "number of time frames := 1\n";
  output_header << "!END OF INTERFILE :=\n";

  return Succeeded::yes;
}

Succeeded
write_basic_interfile_PDFS_header(const string& data_filename,
			    const ProjDataFromStream& pdfs)

{
  char header_file_name[max_filename_length];
  strcpy(header_file_name,data_filename.c_str());
  replace_extension(header_file_name,".hs");
  return 
    write_basic_interfile_PDFS_header(header_file_name,
				    data_filename,pdfs);
}

/**********************************************************************
   template instantiations
   **********************************************************************/


template 
Succeeded 
write_basic_interfile<>(const string&  filename, 
				      const Array<3,signed short>&,
				      const CartesianCoordinate3D<float>& voxel_size,
				      const NumericType output_type);
template 
Succeeded 
write_basic_interfile<>(const string&  filename, 
				      const Array<3,unsigned short>&,
				      const CartesianCoordinate3D<float>& voxel_size,
				      const NumericType output_type);

template 
Succeeded 
write_basic_interfile<>(const string&  filename, 
				      const Array<3,float>&,
				      const CartesianCoordinate3D<float>& voxel_size,
				      const NumericType output_type);


template 
Succeeded 
write_basic_interfile<>(const string& filename, 
		      const Array<3,signed short>& image,
		      const NumericType output_type);

template 
Succeeded 
write_basic_interfile<>(const string& filename, 
		      const Array<3,unsigned short>& image,
		      const NumericType output_type);

template 
Succeeded 
write_basic_interfile<>(const string& filename, 
		      const Array<3,float>& image,
		      const NumericType output_type);

END_NAMESPACE_TOMO
