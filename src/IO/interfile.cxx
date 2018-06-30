/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2011, Hammersmith Imanet Ltd
    Copyright (C) 2013, 2018, University College London
    This file is part of STIR.

    This file is free software; you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation; either version 2.1 of the License, or
    (at your option) any later version.

    This file is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    See STIR/LICENSE.txt for details
*/
/*!
  \file 
  \ingroup InterfileIO
 
  \brief  Implementation of functions which read/write Interfile data

  \author Kris Thielemans 
  \author Sanida Mustafovic
  \author PARAPET project
*/
//   Pretty horrible implementations at the moment...


#include "stir/ExamInfo.h"
#include "stir/TimeFrameDefinitions.h"
#include "stir/PatientPosition.h"
#include "stir/ImagingModality.h"
#include "stir/IO/interfile.h"
#include "stir/interfile_keyword_functions.h"
#include "stir/IO/InterfileHeader.h"
#include "stir/IO/InterfileHeaderSiemens.h"
#include "stir/IO/InterfilePDFSHeaderSPECT.h"
#include "stir/IndexRange3D.h"
#include "stir/utilities.h"
#include "stir/CartesianCoordinate3D.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/ProjDataFromStream.h"
#include "stir/ProjDataInfoCylindricalArcCorr.h"
#include "stir/Scanner.h"
#include "stir/Succeeded.h"
#include "stir/IO/write_data.h"
#include "stir/IO/read_data.h"
#include "stir/is_null_ptr.h"
#include "stir/Bin.h"
#include "stir/stream.h"
#include "stir/error.h"
#include <boost/format.hpp>
#include <fstream>
#include <algorithm>

#ifndef STIR_NO_NAMESPACES
using std::cerr;
using std::endl;
using std::ofstream;
using std::ifstream;
using std::iostream;
using std::fstream;
using std::string;
using std::vector;
using std::istream;
using std::string;
using std::ios;
#endif

START_NAMESPACE_STIR

bool 
is_interfile_signature(const char * const signature)
{
  // checking for "interfile :"
  const char * pos_of_colon = strchr(signature, ':');
  if (pos_of_colon == NULL)
    return false;
  string keyword(signature, pos_of_colon-signature);
  return (
	  standardise_interfile_keyword(keyword) == 
	  standardise_interfile_keyword("interfile"));
}



VoxelsOnCartesianGrid<float>* 
read_interfile_image(istream& input, 
		     const string&  directory_for_data)
{
  InterfileImageHeader hdr;
  if (!hdr.parse(input))
  {
      return 0; //KT 10/12/2001 do not call ask_parameters anymore 
    }
  
  // prepend directory_for_data to the data_file_name from the header
  char full_data_file_name[max_filename_length];
  strcpy(full_data_file_name, hdr.data_file_name.c_str());
  prepend_directory_name(full_data_file_name, directory_for_data.c_str());  
  
  ifstream data_in;
  open_read_binary(data_in, full_data_file_name);
    
  CartesianCoordinate3D<float> voxel_size(static_cast<float>(hdr.pixel_sizes[2]), 
					  static_cast<float>(hdr.pixel_sizes[1]), 
					  static_cast<float>(hdr.pixel_sizes[0]));
  
  const int z_size =  hdr.matrix_size[2][0];
  const int y_size =  hdr.matrix_size[1][0];
  const int x_size =  hdr.matrix_size[0][0];
  const BasicCoordinate<3,int> min_indices = 
    make_coordinate(0, -y_size/2, -x_size/2);
  const BasicCoordinate<3,int> max_indices = 
    min_indices + make_coordinate(z_size, y_size, x_size) - 1;

  CartesianCoordinate3D<float> origin(0,0,0);
  if (hdr.first_pixel_offsets[2] != InterfileHeader::double_value_not_set)
    {
      // make sure that origin is such that 
      // first_pixel_offsets =  min_indices*voxel_size + origin
      origin =
	make_coordinate(float(hdr.first_pixel_offsets[2]),
			float(hdr.first_pixel_offsets[1]),
			float(hdr.first_pixel_offsets[0]))
	- voxel_size * BasicCoordinate<3,float>(min_indices);
    }

  VoxelsOnCartesianGrid<float>* image_ptr =
    new VoxelsOnCartesianGrid<float>
    (hdr.get_exam_info_sptr(),
     IndexRange<3>(min_indices, max_indices),
     origin,
     voxel_size);
  
  data_in.seekg(hdr.data_offset_each_dataset[0]);
  
  float scale = float(1);
  if (read_data(data_in, *image_ptr, hdr.type_of_numbers, scale, hdr.file_byte_order)
    == Succeeded::no
    || scale != 1)
  {
    warning("read_interfile_image: error reading data or scale factor returned by read_data not equal to 1\n");
    return 0;
  }
  
  for (int i=0; i< hdr.matrix_size[2][0]; i++)
    if (hdr.image_scaling_factors[0][i]!= 1)
      (*image_ptr)[i] *= static_cast<float>(hdr.image_scaling_factors[0][i]);
    
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

#if 0
const VectorWithOffset<std::streamoff> 
compute_file_offsets(int number_of_time_frames,
		     const NumericType output_type,
		     const CartesianCoordinate3D<int>& dim,
		     std::streamoff initial_offset)
{ 
  const std::size_t a= dim.x()*dim.y()*dim.z()*output_type.size_in_bytes();
  VectorWithOffset<std::streamoff> temp(number_of_time_frames);
  {
    for (int i=0; i<=number_of_time_frames-1;i++)
      temp[i]= initial_offset +i*a;
  }
  return temp;
}
#endif
/* A function that finds the appropriate filename for the binary data
   to write in the header. It tries to cut the directory part of
   data_file_name if it's the same as the directory part of the header.
*/
static
string 
interfile_get_data_file_name_in_header(const string& header_file_name, 
 				    const string& data_file_name)
{
  const string dir_name_of_binary_data =
    get_directory_name(data_file_name);
  if (dir_name_of_binary_data.size() == 0 ||
      is_absolute_pathname(data_file_name))
    {
      // data_dirname is empty or it's an absolute path
      return data_file_name;
    }
  const string dir_name_of_header =
    get_directory_name(header_file_name);
  if (dir_name_of_header == dir_name_of_binary_data)
    {
      // dirnames are the same, so strip from data_file_name
      return
        string(data_file_name,
               find_pos_of_filename(data_file_name), 
                string::npos);
    }
  else
    {
      // just copy, what else to do?
      return data_file_name;
    }
}

//// some static helper functions for writing
// probably should be moved to InterfileHeader
static void write_interfile_patient_position(std::ostream& output_header, const ExamInfo& exam_info)
{
  string orientation;
  switch (exam_info.patient_position.get_orientation())
    {
    case PatientPosition::head_in: orientation="head_in";break;
    case PatientPosition::feet_in: orientation="feet_in";break;
    case PatientPosition::other_orientation: orientation="other";break;
    default: orientation="unknown"; break;
    }
  string rotation;
  switch (exam_info.patient_position.get_rotation())
    {
    case PatientPosition::prone: rotation="prone";break;
    case PatientPosition::supine: rotation="supine";break;
    case PatientPosition::other_rotation:
    case PatientPosition::left:
    case PatientPosition::right:
      rotation="other";break;
    default: rotation="unknown"; break;
    }
  if (orientation!="unknown")
    output_header << "patient orientation := " << orientation << '\n';
  if (rotation!="unknown")
    output_header << "patient rotation := " << rotation << '\n';
}

static void write_interfile_time_frame_definitions(std::ostream& output_header, const ExamInfo& exam_info)
  // TODO this is according to the proposed interfile standard for PET. Interfile 3.3 is different
{
  const TimeFrameDefinitions& frame_defs(exam_info.time_frame_definitions);
  if (frame_defs.get_num_time_frames()>0)
    {
      output_header << "number of time frames := " << frame_defs.get_num_time_frames() << '\n';
      for (unsigned int frame_num=1; frame_num<=frame_defs.get_num_time_frames(); ++frame_num)
        {
          if (frame_defs.get_duration(frame_num)>0)
            {
              output_header << "image duration (sec)[" << frame_num << "] := "
                            << frame_defs.get_duration(frame_num) << '\n';
              output_header << "image relative start time (sec)[" << frame_num << "] := "
                            << frame_defs.get_start_time(frame_num) << '\n';
            }
        }
    }
  else
    {
      // need to write this anyway to allow vectored keys below
      output_header << "number of time frames := 1\n";
    }
}

// Write energy window lower and upper thresholds, if they are not -1
static void write_interfile_energy_windows(std::ostream& output_header, const ExamInfo& exam_info)
{
  if (exam_info.get_high_energy_thres() > 0 &&
      exam_info.get_low_energy_thres() >= 0)
    {
      output_header << "energy window lower level := " <<
        exam_info.get_low_energy_thres() << '\n';
      output_header << "energy window upper level :=  " <<
        exam_info.get_high_energy_thres() << '\n';
    }
}

static void  write_interfile_modality(std::ostream& output_header, const ExamInfo& exam_info)
{
  /*
  // default modality is PET
  ImagingModality imaging_modality(ImagingModality::PT);
  if (!is_null_ptr(exam_info_ptr))
    imaging_modality=exam_info_ptr->imaging_modality;
  */
  if (exam_info.
      imaging_modality.get_modality() != ImagingModality::Unknown)
    output_header << "!imaging modality := " << exam_info.imaging_modality.get_name() << '\n';
}
////// end static functions

Succeeded 
write_basic_interfile_image_header(const string& header_file_name,
				   const string& image_file_name,
                                   const ExamInfo& exam_info,
				   const IndexRange<3>& index_range,
				   const CartesianCoordinate3D<float>& voxel_size,
				   const CartesianCoordinate3D<float>& origin,
				   const NumericType output_type,
				   const ByteOrder byte_order,
				   const VectorWithOffset<float>& scaling_factors,
				   const VectorWithOffset<unsigned long>& file_offsets)
{
  CartesianCoordinate3D<int> min_indices;
  CartesianCoordinate3D<int> max_indices;
  if (!index_range.get_regular_range(min_indices, max_indices))
  {
    warning("write_basic_interfile: can handle only regular index ranges\n. No output\n");
    return Succeeded::no;
  }
  CartesianCoordinate3D<int> dimensions = max_indices - min_indices + 1;
  string header_name = header_file_name;
  add_extension(header_name, ".hv");
  ofstream output_header(header_name.c_str(), ios::out);
  if (!output_header.good())
    {
      warning("Error opening Interfile header '%s' for writing\n",
	      header_name.c_str());
      return Succeeded::no;
    }  
 
  // handle directory names
  const string data_file_name_in_header =
    interfile_get_data_file_name_in_header(header_file_name, image_file_name);
 
  output_header << "!INTERFILE  :=\n";
  output_header << "name of data file := " << data_file_name_in_header << endl;
  output_header << "!GENERAL DATA :=\n";
  if (!exam_info.originating_system.empty())
    output_header << "originating system := "
                  << exam_info.originating_system << endl;
  write_interfile_modality(output_header, exam_info);
  write_interfile_patient_position(output_header, exam_info);
  output_header << "!GENERAL IMAGE DATA :=\n";
  output_header << "!type of data := PET\n";
  output_header << "imagedata byte order := " <<
    (byte_order == ByteOrder::little_endian 
     ? "LITTLEENDIAN"
     : "BIGENDIAN")
		<< endl;
  output_header << "!PET STUDY (General) :=\n";
  write_interfile_time_frame_definitions(output_header, exam_info);
  write_interfile_energy_windows(output_header, exam_info);
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

  if (origin.z() != InterfileHeader::double_value_not_set)
    {
      const CartesianCoordinate3D<float> first_pixel_offsets =
	voxel_size * BasicCoordinate<3,float>(min_indices) + origin;
      output_header << "first pixel offset (mm) [1] := " 
		    << first_pixel_offsets.x() << '\n';
      output_header << "first pixel offset (mm) [2] := " 
		    << first_pixel_offsets.y() << '\n';
      output_header << "first pixel offset (mm) [3] := " 
		    << first_pixel_offsets.z() << '\n';
    }
  
  for (int i=1; i<=scaling_factors.get_length();i++)
    {
      // only write scaling factors and offset if more than 1 frame or they are not default values
      if (scaling_factors[i-1]!=1. || scaling_factors.get_length()>1)
	output_header << "image scaling factor"<<"["<<i<<"] := " << scaling_factors[i-1]<< endl;
      if (file_offsets[i-1] || scaling_factors.get_length()>1)
	output_header << "data offset in bytes"<<"["<<i<<"] := "<< file_offsets[i-1]<< endl;
    }

  // analogue of image scaling factor[*] for Louvain la Neuve Interfile reader
  { 
    bool output_quantification_units = true;
    if (scaling_factors.get_length() > 1)
    {
      const float first_scaling_factor = scaling_factors[0];
      for (int i=1; i <= scaling_factors.get_max_index(); i++)
	{
	  if (scaling_factors[i] != first_scaling_factor)
	    {
	      warning("Interfile: non-standard 'quantification units' keyword not set as not all scale factors are the same");
              output_quantification_units = false;
	      break;
	    }
	}
    }
    if (output_quantification_units)
    {
      // only write when not 1
      output_quantification_units = (scaling_factors[0] != 1.);
    }
    if (output_quantification_units)
      output_header << "quantification units := " << scaling_factors[0] << endl;
  }
  // TODO
  // output_header << "maximum pixel count := " << image.find_max()/scale << endl;  
  output_header << "!END OF INTERFILE :=\n";

  
  // temporary copy to make an old-style header to satisfy Analyze
  {
    string header_name = header_file_name;
    replace_extension(header_name, ".ahv");

    ofstream output_header(header_name.c_str(), ios::out);
    if (!output_header.good())
      {
	error(boost::format("Error opening old-style Interfile header %1% for writing") % header_name);
        return Succeeded::no;
      }  
    
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
      output_header << (output_type.size_in_bytes()==4 
                        ? "short float\n" : "long float\n");
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
      output_header << ";Correct value is of keyword (commented out)\n"
                    << ";!slice thickness (pixels) := " 
                    << voxel_size.z()/ voxel_size.x()<< endl;
      output_header << ";Value for Analyze\n"
                    << "!slice thickness (pixels) := " 
		    << zsize << endl;
    }
    output_header << "!END OF INTERFILE :=\n";
    
  }
  return Succeeded::yes;
}



#if !defined(_MSC_VER) || (_MSC_VER > 1300)
template <class elemT>
#else
#define elemT float
#endif
Succeeded 
write_basic_interfile(const string& filename, 
		      const Array<3,elemT>& image,
		      const NumericType output_type,
		      const float scale,
		      const ByteOrder byte_order)
{
  CartesianCoordinate3D<float> origin;
  origin.fill(static_cast<float>(InterfileHeader::double_value_not_set));
  return
    write_basic_interfile(filename, 
			  image, 
			  CartesianCoordinate3D<float>(1,1,1), 
			  origin,
			  output_type,
			  scale,
			  byte_order);
}
#if defined(_MSC_VER) && (_MSC_VER <= 1300)
#undef elemT 
#endif

template <class NUMBER>
Succeeded write_basic_interfile(const string&  filename,
                                const ExamInfo& exam_info,
				const Array<3,NUMBER>& image,
				const CartesianCoordinate3D<float>& voxel_size,
				const CartesianCoordinate3D<float>& origin,
				const NumericType output_type,
				const float scale,
				const ByteOrder byte_order)
{

#if 1
  string data_name=filename;
  {
    string::size_type pos=find_pos_of_extension(filename);
    if (pos!=string::npos && filename.substr(pos)==".hv")
      replace_extension(data_name, ".v");
    else
      add_extension(data_name, ".v");
  }
  string header_name=filename;
#else
  char * data_name = new char[filename.size() + 5];
  char * header_name = new char[filename.size() + 5];

  strcpy(data_name, filename.c_str());
  // KT 29/06/2001 make sure that a filename ending on .hv is treated correctly
  {
   const char * const extension = strchr(find_filename(data_name),'.');
   if (extension!=NULL && strcmp(extension, ".hv")==0)
     replace_extension(data_name, ".v");
   else
     add_extension(data_name, ".v");
  }
  strcpy(header_name, data_name);
#endif
  replace_extension(header_name, ".hv");

  ofstream output_data;
  open_write_binary(output_data, data_name.c_str());

  float scale_to_use = scale;
  write_data(output_data, image, output_type, scale_to_use,
	     byte_order);

  VectorWithOffset<float> scaling_factors(1);
  scaling_factors[0] = scale_to_use;
  VectorWithOffset<unsigned long> file_offsets(1);
  file_offsets.fill(0);


  const Succeeded success =
    write_basic_interfile_image_header(header_name,
				       data_name,
                                       exam_info,
				       image.get_index_range(), 
				       voxel_size,
				       origin,
				       output_type,
				       byte_order,
				       scaling_factors,
				       file_offsets);
#if 0
  delete[] header_name;
  delete[] data_name;
#endif
  return success;
}

template <class NUMBER>
Succeeded write_basic_interfile(const string&  filename, 
				const Array<3,NUMBER>& image,
				const CartesianCoordinate3D<float>& voxel_size,
				const CartesianCoordinate3D<float>& origin,
				const NumericType output_type,
				const float scale,
				const ByteOrder byte_order)
{

  return write_basic_interfile(filename,
                               ExamInfo(),
                               image,
                               voxel_size,
                               origin,
                               output_type,
                               scale,
                               byte_order);
}

Succeeded
write_basic_interfile(const string&  filename, 
		      const VoxelsOnCartesianGrid<float>& image,
		      const NumericType output_type,
		      const float scale,
		      const ByteOrder byte_order)
{
  return
    write_basic_interfile(filename,
                          image.get_exam_info(),
			  image, // use automatic reference to base class
			  image.get_grid_spacing(), 
			  image.get_origin(),
			  output_type,
			  scale,
			  byte_order);
}

Succeeded 
write_basic_interfile(const string& filename, 
		      const DiscretisedDensity<3,float>& image,
		      const NumericType output_type,
		      const float scale,
		      const ByteOrder byte_order)
{
  // dynamic_cast will throw an exception when it's not valid
  return
    write_basic_interfile(filename,
                          dynamic_cast<const VoxelsOnCartesianGrid<float>& >(image),
                          output_type,
			  scale, byte_order);
}


static ProjDataFromStream* 
read_interfile_PDFS_SPECT(istream& input,
		    const string& directory_for_data,
		    const ios::openmode open_mode)
{
  
  InterfilePDFSHeaderSPECT hdr;  
   if (!hdr.parse(input))
    {
      return 0; // KT 10122001 do not call ask_parameters anymore
    }

  char full_data_file_name[max_filename_length];
  strcpy(full_data_file_name, hdr.data_file_name.c_str());
  prepend_directory_name(full_data_file_name, directory_for_data.c_str());  
 
  vector<int> segment_sequence(1);  
  segment_sequence[0]=0;
   
  for (unsigned int i=1; i<hdr.image_scaling_factors[0].size(); i++)
    if (hdr.image_scaling_factors[0][0] != hdr.image_scaling_factors[0][i])
      { 
	error("Interfile error: all image scaling factors should be equal at the moment.");
      }
  
  assert(!is_null_ptr(hdr.data_info_sptr));

   shared_ptr<iostream> data_in(new fstream (full_data_file_name, open_mode | ios::binary));
   if (!data_in->good())
     {
       warning("interfile parsing: error opening file %s",full_data_file_name);
       return 0;
     }

   return new ProjDataFromStream(hdr.get_exam_info_sptr(), 
				 hdr.data_info_sptr,
				 data_in,
				 hdr.data_offset_each_dataset[0],
				 segment_sequence,
				 hdr.storage_order,
				 hdr.type_of_numbers,
				 hdr.file_byte_order,
				 static_cast<float>(hdr.image_scaling_factors[0][0]));


}


ProjDataFromStream*
read_interfile_PDFS_Siemens(istream& input,
  const string& directory_for_data,
  const ios::openmode open_mode)
{
  InterfilePDFSHeaderSiemens hdr;
  if (!hdr.parse(input))
    {
      warning("Interfile parsing of Siemens Interfile projection data failed");
      return 0;
    }
  // KT 14/01/2000 added directory capability
  // prepend directory_for_data to the data_file_name from the header

  char full_data_file_name[max_filename_length];
  strcpy(full_data_file_name, hdr.data_file_name.c_str());
  prepend_directory_name(full_data_file_name, directory_for_data.c_str());

  shared_ptr<iostream> data_in(new fstream(full_data_file_name, open_mode | ios::binary));
  if (!data_in->good())
    {
    warning("interfile parsing: error opening file %s", full_data_file_name);
    return 0;
    }

  if (hdr.compression)
    warning("Siemens projection data is compressed. Reading of raw data will fail.");

  return new ProjDataFromStream(hdr.get_exam_info_sptr(),
    hdr.data_info_ptr->create_shared_clone(),
    data_in,
    hdr.data_offset_each_dataset[0],
    hdr.segment_sequence,
    hdr.storage_order,
    hdr.type_of_numbers,
    hdr.file_byte_order,
    1.);

}

ProjDataFromStream* 
read_interfile_PDFS(istream& input,
		    const string& directory_for_data,
		    const ios::openmode open_mode)
{
  
  {
    MinimalInterfileHeader hdr;  
    std::ios::off_type offset = input.tellg();
    if (!hdr.parse(input, false)) // parse without warnings
      {
        warning("Interfile parsing failed");
        return 0;
      }
    input.seekg(offset);
    if (hdr.get_exam_info_ptr()->imaging_modality.get_modality() ==
        ImagingModality::NM)
      {
        // spect data
        input.seekg(offset);
        return read_interfile_PDFS_SPECT(input, directory_for_data, open_mode); 
      }
	  if (!hdr.siemens_mi_version.empty())
      {
		     input.seekg(offset);
         return read_interfile_PDFS_Siemens(input, directory_for_data, open_mode);
      }
	}
    
  // if we get here, it's PET

  InterfilePDFSHeader hdr;  
  if (!hdr.parse(input))
    {
      warning("Interfile parsing of PET projection data failed");
      return 0;
    }

  // KT 14/01/2000 added directory capability
  // prepend directory_for_data to the data_file_name from the header

  char full_data_file_name[max_filename_length];
  strcpy(full_data_file_name, hdr.data_file_name.c_str());
  prepend_directory_name(full_data_file_name, directory_for_data.c_str());  
  
  for (unsigned int i=1; i<hdr.image_scaling_factors[0].size(); i++)
    if (hdr.image_scaling_factors[0][0] != hdr.image_scaling_factors[0][i])
      { 
	warning("Interfile warning: all image scaling factors should be equal \n"
		"at the moment. Using the first scale factor only.\n");
	break;
      }
  
   assert(hdr.data_info_ptr !=0);

   shared_ptr<iostream> data_in(new fstream (full_data_file_name, open_mode | ios::binary));
   if (!data_in->good())
     {
       warning("interfile parsing: error opening file %s",full_data_file_name);
       return 0;
     }

   return new ProjDataFromStream(hdr.get_exam_info_sptr(),
				 hdr.data_info_ptr->create_shared_clone(),
				 data_in,
				 hdr.data_offset_each_dataset[0],
				 hdr.segment_sequence,
				 hdr.storage_order,
				 hdr.type_of_numbers,
				 hdr.file_byte_order,
				 static_cast<float>(hdr.image_scaling_factors[0][0]));


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

Succeeded 
write_basic_interfile_PDFS_header(const string& header_file_name,
				  const string& data_file_name,
				  const ProjDataFromStream& pdfs)
{

  string header_name = header_file_name;
  add_extension(header_name, ".hs");
  ofstream output_header(header_name.c_str(), ios::out);
  if (!output_header.good())
    {
      warning("Error opening Interfile header '%s' for writing\n",
	      header_name.c_str());
      return Succeeded::no;
    }  

  // handle directory names
  const string data_file_name_in_header =
    interfile_get_data_file_name_in_header(header_file_name, data_file_name);

  const vector<int> segment_sequence = pdfs.get_segment_sequence_in_stream();

#if 0
  // TODO get_phi currently ignores view offset
  const float angle_first_view = 
    pdfs.get_proj_data_info_ptr()->get_phi(Bin(0,0,0,0)) * float(180/_PI);
#else
  const float angle_first_view = 
    pdfs.get_proj_data_info_ptr()->get_scanner_ptr()->get_default_intrinsic_tilt() * float(180/_PI);
#endif
  const float angle_increment = 
    (pdfs.get_proj_data_info_ptr()->get_phi(Bin(0,1,0,0)) -
     pdfs.get_proj_data_info_ptr()->get_phi(Bin(0,0,0,0))) * float(180/_PI);

  output_header << "!INTERFILE  :=\n";

  const bool is_spect = pdfs.get_exam_info().imaging_modality.get_modality() == ImagingModality::NM;

  write_interfile_modality(output_header, pdfs.get_exam_info());

  output_header << "name of data file := " << data_file_name_in_header << endl;

  output_header << "originating system := ";
  output_header <<pdfs.get_proj_data_info_ptr()->get_scanner_ptr()->get_name() << endl;

  if (is_spect)
    output_header << "!version of keys := 3.3\n";
  else
    output_header << "!version of keys := STIR3.0\n";

  output_header << "!GENERAL DATA :=\n";
  output_header << "!GENERAL IMAGE DATA :=\n";
  output_header << "!type of data := " << (is_spect ? "Tomographic" : "PET") << '\n';

  // output patient position
  // note: strictly speaking this should come after "!SPECT STUDY (general)" but 
  // that's strange as these keys would be useful for all other cases as well
  write_interfile_patient_position(output_header, pdfs.get_exam_info());

  output_header << "imagedata byte order := " <<
    (pdfs.get_byte_order_in_stream() == ByteOrder::little_endian 
     ? "LITTLEENDIAN"
     : "BIGENDIAN")
		<< endl;

  
  if (is_spect)
    {
      // output_header << "number of energy windows :=1\n";
      output_header << "!SPECT STUDY (General) :=\n";
    }
  else
    {
      output_header << "!PET STUDY (General) :=\n";
      output_header << "!PET data type := Emission\n";
      {
        // KT 10/12/2001 write applied corrections keyword
        if(dynamic_cast< const ProjDataInfoCylindricalArcCorr*> (pdfs.get_proj_data_info_ptr()) != NULL)
          output_header << "applied corrections := {arc correction}\n";
        else
          output_header << "applied corrections := {None}\n";
      }
    }
  {
    string number_format;
    size_t size_in_bytes;

    const NumericType data_type = pdfs.get_data_type_in_stream();
    data_type.get_Interfile_info(number_format, size_in_bytes);

    output_header << "!number format := " << number_format << "\n";
    output_header << "!number of bytes per pixel := " << size_in_bytes << "\n";
  }

  if (is_spect)
    {
      output_header << "!number of projections := " << pdfs.get_num_views() << '\n';
      output_header << "!extent of rotation := " << pdfs.get_num_views() * fabs(angle_increment) << '\n';        
      output_header << "process status := acquired\n";
      output_header << "!SPECT STUDY (acquired data):=\n";

      output_header << "!direction of rotation := " 
                    <<  (angle_increment>0 ? "CCW" : "CW")
                    << '\n';
      output_header << "start angle := " << angle_first_view << '\n';

      ProjDataInfoCylindricalArcCorr const * proj_data_info_cyl_ptr = 
        dynamic_cast<ProjDataInfoCylindricalArcCorr const *>(pdfs.get_proj_data_info_ptr());

      VectorWithOffset<float> ring_radii = proj_data_info_cyl_ptr->get_ring_radii_for_all_views();
      if (*std::min_element(ring_radii.begin(),ring_radii.end()) == 
          *std::max_element(ring_radii.begin(),ring_radii.end()))
        {
          output_header << "orbit := Circular\n";
          output_header << "Radius := " << *ring_radii.begin() << '\n';
        }
      else
        {
          output_header << "orbit := Non-circular\n";
          output_header << "Radius := " << ring_radii << '\n';
        }

      output_header << "!matrix size [1] := "
                    <<proj_data_info_cyl_ptr->get_num_tangential_poss() << '\n';
      output_header << "!scaling factor (mm/pixel) [1] := "
                    <<proj_data_info_cyl_ptr->get_tangential_sampling() << '\n';
      output_header << "!matrix size [2] := "
                    <<proj_data_info_cyl_ptr->get_num_axial_poss(0) << '\n';
      output_header << "!scaling factor (mm/pixel) [2] := "
                    <<proj_data_info_cyl_ptr->get_axial_sampling(0) << '\n';

      if (pdfs.get_offset_in_stream())
	output_header<<"data offset in bytes := "
		     <<pdfs.get_offset_in_stream()<<'\n';;
      output_header << "!END OF INTERFILE :=\n";

      return Succeeded::yes;
    }

  // it's PET data if we get here

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
	  error("write_interfile_PSOV_header: unsupported storage order, "
                "defaulting to Segment_View_AxialPos_TangPos.\n Please correct by hand !");
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
      std::vector<int>::const_iterator seg = segment_sequence.begin();
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
	 std::vector<int>::const_iterator seg = segment_sequence.begin();
	 output_header << "{ " << proj_data_info_ptr->get_min_ring_difference(*seg);
	 for (seg++; seg != segment_sequence.end(); seg++)
	   output_header << "," <<proj_data_info_ptr->get_min_ring_difference(*seg);
	 output_header << "}\n";
       }

       output_header << "maximum ring difference per segment := ";
       {
	 std::vector<int>::const_iterator seg = segment_sequence.begin();
	 output_header << "{ " <<proj_data_info_ptr->get_max_ring_difference(*seg);
	 for (seg++; seg != segment_sequence.end(); seg++)
	   output_header << "," <<proj_data_info_ptr->get_max_ring_difference(*seg);
	 output_header << "}\n";
       }
  
       const Scanner& scanner = *proj_data_info_ptr->get_scanner_ptr();
       if (fabs(proj_data_info_ptr->get_ring_radius()-
		scanner.get_effective_ring_radius()) > .1)
	 warning("write_basic_interfile_PDFS_header: inconsistent effective ring radius:\n"
		 "\tproj_data_info has %g, scanner has %g.\n"
		 "\tThis really should not happen and signifies a bug.\n"
		 "\tYou will have a problem reading this data back in.",
		 proj_data_info_ptr->get_ring_radius(),
		 scanner.get_effective_ring_radius());
       if (fabs(proj_data_info_ptr->get_ring_spacing()-
		scanner.get_ring_spacing()) > .1)
	 warning("write_basic_interfile_PDFS_header: inconsistent ring spacing:\n"
		 "\tproj_data_info has %g, scanner has %g.\n"
		 "\tThis really should not happen and signifies a bug.\n"
		 "\tYou will have a problem reading this data back in.",
		 proj_data_info_ptr->get_ring_spacing(),
		 scanner.get_ring_spacing());

       output_header << scanner.parameter_info();

       output_header << "effective central bin size (cm) := " 
		     << proj_data_info_ptr->get_sampling_in_s(Bin(0,0,0,0))/10. << endl;

     } // end of cylindrical scanner
  else
    {
      // TODO something here
    }


   // write time frame info and energy windows
   write_interfile_time_frame_definitions(output_header, pdfs.get_exam_info());
   write_interfile_energy_windows(output_header, pdfs.get_exam_info());

  if (pdfs.get_scale_factor()!=1.F)
 output_header <<"image scaling factor[1] := "
		<<pdfs.get_scale_factor()<<endl;

  if (pdfs.get_offset_in_stream())
    output_header<<"data offset in bytes[1] := "
                 <<pdfs.get_offset_in_stream()<<endl;
    
   output_header << "!END OF INTERFILE :=\n";

  return Succeeded::yes;
}

Succeeded
write_basic_interfile_PDFS_header(const string& data_filename,
			    const ProjDataFromStream& pdfs)

{
  // KT 26/08/2001 make sure that a data_filename ending on .hs is treated correctly
#if 1
  string header_file_name = data_filename;
  string new_data_file_name = data_filename;
  {
    string::size_type pos=find_pos_of_extension(data_filename);
    if (pos!=string::npos && data_filename.substr(pos)==".hs")
      replace_extension(new_data_file_name, ".s");
    else
      add_extension(new_data_file_name, ".s");
  }
#else
  char header_file_name[max_filename_length];
  char new_data_file_name[max_filename_length];
  strcpy(header_file_name,data_filename.c_str());
  strcpy(new_data_file_name,data_filename.c_str());
  {
   const char * const extension = strchr(find_filename(new_data_file_name),'.');
   if (extension!=NULL && strcmp(extension, ".hs")==0)
     replace_extension(new_data_file_name, ".s");
   else
     add_extension(new_data_file_name, ".s");
  }
#endif

  replace_extension(header_file_name,".hs");

  return 
    write_basic_interfile_PDFS_header(header_file_name,
				      new_data_file_name,
				      pdfs);
}

/**********************************************************************
   template instantiations
   **********************************************************************/


template 
Succeeded 
write_basic_interfile<>(const string&  filename, 
			const Array<3,signed short>&,
			const CartesianCoordinate3D<float>& voxel_size,
			const CartesianCoordinate3D<float>& origin,
			const NumericType output_type,
			const float scale,
			const ByteOrder byte_order);
template 
Succeeded 
write_basic_interfile<>(const string&  filename, 
			const Array<3,unsigned short>&,
			const CartesianCoordinate3D<float>& voxel_size,
			const CartesianCoordinate3D<float>& origin,
			const NumericType output_type,
			const float scale,
			const ByteOrder byte_order);

template 
Succeeded 
write_basic_interfile<>(const string&  filename, 
			const Array<3,float>&,
			const CartesianCoordinate3D<float>& voxel_size,
			const CartesianCoordinate3D<float>& origin,
			const NumericType output_type,
			const float scale,
			const ByteOrder byte_order);

#if !defined(_MSC_VER) || (_MSC_VER > 1300)

template 
Succeeded 
write_basic_interfile<>(const string& filename, 
		      const Array<3,signed short>& image,
		      const NumericType output_type,
			const float scale,
			const ByteOrder byte_order);

template 
Succeeded 
write_basic_interfile<>(const string& filename, 
		      const Array<3,unsigned short>& image,
		      const NumericType output_type,
			const float scale,
			const ByteOrder byte_order);

template 
Succeeded 
write_basic_interfile<>(const string& filename, 
		      const Array<3,float>& image,
		      const NumericType output_type,
			const float scale,
			const ByteOrder byte_order);
#endif

END_NAMESPACE_STIR
