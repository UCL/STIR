//
// $Id$
//
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, Hammersmith Imanet Ltd
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
  \brief implementations for the stir::InterfileHeader class

  \author Kris Thielemans
  \author PARAPET project

  $Date$
  $Revision$

*/

#include "stir/IO/InterfileHeader.h"
#include "stir/ProjDataInfoCylindricalArcCorr.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include <numeric>
#include <functional>

#ifndef STIR_NO_NAMESPACES
using std::binary_function;
using std::pair;
using std::sort;
using std::cerr;
using std::endl;
#endif

START_NAMESPACE_STIR
InterfileHeader::InterfileHeader()
     : KeyParser()
{

  number_format_values.push_back("bit");
  number_format_values.push_back("ascii");
  number_format_values.push_back("signed integer");
  number_format_values.push_back("unsigned integer");
  number_format_values.push_back("float");
  
  byte_order_values.push_back("LITTLEENDIAN");
  byte_order_values.push_back("BIGENDIAN");
  
  PET_data_type_values.push_back("Emission");
  PET_data_type_values.push_back("Transmission");
  PET_data_type_values.push_back("Blank");
  PET_data_type_values.push_back("AttenuationCorrection");
  PET_data_type_values.push_back("Normalisation");
  PET_data_type_values.push_back("Image");
  
  type_of_data_values.push_back("Static");
  type_of_data_values.push_back("Dynamic");
  type_of_data_values.push_back("Tomographic");
  type_of_data_values.push_back("Curve");
  type_of_data_values.push_back("ROI");
  type_of_data_values.push_back("PET");
  type_of_data_values.push_back("Other");
  
  patient_orientation_values.push_back("head_in"); //default
  patient_orientation_values.push_back("feet_in");
  patient_orientation_values.push_back("other");

  patient_rotation_values.push_back("supine"); //default
  patient_rotation_values.push_back("prone");
  patient_rotation_values.push_back("other");

  // default values
  // KT 07/10/2002 added 2 new ones
  number_format_index = 3; // unsigned integer
  bytes_per_pixel = -1; // standard does not provide a default
  // KT 02/11/98 set default for correct variable
  byte_order_index = 1;//  file_byte_order = ByteOrder::big_endian;
  type_of_data_index = 6; // PET
  PET_data_type_index = 5; // Image
  patient_orientation_index = 0; //head-in
  patient_rotation_index = 0; //supine
  num_dimensions = 0;
  num_time_frames = 1;
  image_scaling_factors.resize(num_time_frames);
  for (int i=0; i<num_time_frames; i++)
    image_scaling_factors[i].resize(1, 1.);
  lln_quantification_units = 1.;

  data_offset.resize(num_time_frames, 0UL);


  // KT 09/10/98 replaced NULL arguments with the do_nothing function
  // as gcc cannot convert 0 to a 'pointer to member function'  
  add_key("INTERFILE", 
    KeyArgument::NONE,	&KeyParser::start_parsing);
  add_key("name of data file", 
    KeyArgument::ASCII,	&data_file_name);
  add_key("originating system",
    KeyArgument::ASCII, &originating_system);
  add_key("GENERAL DATA", 
    KeyArgument::NONE,	&KeyParser::do_nothing);
  add_key("GENERAL IMAGE DATA", 
    KeyArgument::NONE,	&KeyParser::do_nothing);
  add_key("type of data", 
    KeyArgument::ASCIIlist,
    &type_of_data_index, 
    &type_of_data_values);

  add_key("patient orientation",
	  KeyArgument::ASCIIlist,
	  &patient_orientation_index,
	  &patient_orientation_values);
  add_key("patient rotation_index",
	  KeyArgument::ASCIIlist,
	  &patient_rotation_index,
	  &patient_rotation_values);


  add_key("imagedata byte order", 
    KeyArgument::ASCIIlist,
    &byte_order_index, 
    &byte_order_values);
  add_key("PET STUDY (General)", 
    KeyArgument::NONE,	&KeyParser::do_nothing);
  add_key("PET data type", 
    KeyArgument::ASCIIlist,
    &PET_data_type_index, 
    &PET_data_type_values);
  
  add_key("data format", 
    KeyArgument::ASCII,	&KeyParser::do_nothing);
  add_key("number format", 
    KeyArgument::ASCIIlist,
    &number_format_index,
    &number_format_values);
  add_key("number of bytes per pixel", 
    KeyArgument::INT,	&bytes_per_pixel);
  add_key("number of dimensions", 
    KeyArgument::INT,	(KeywordProcessor)&InterfileHeader::read_matrix_info,&num_dimensions);
  add_key("matrix size", 
    KeyArgument::LIST_OF_INTS,&matrix_size);
  add_key("matrix axis label", 
    KeyArgument::ASCII,	&matrix_labels);
  add_key("scaling factor (mm/pixel)", 
    KeyArgument::DOUBLE, &pixel_sizes);
  add_key("number of time frames", 
    KeyArgument::INT,	(KeywordProcessor)&InterfileHeader::read_frames_info,&num_time_frames);
  add_key("PET STUDY (Emission data)", 
    KeyArgument::NONE,	&KeyParser::do_nothing);
  add_key("PET STUDY (Image data)", 
    KeyArgument::NONE,	&KeyParser::do_nothing);
  // TODO
  add_key("process status", 
    KeyArgument::NONE,	&KeyParser::do_nothing);
  add_key("IMAGE DATA DESCRIPTION", 
    KeyArgument::NONE,	&KeyParser::do_nothing);

  //TODO
  add_key("maximum pixel count", 
    KeyArgument::NONE,	&KeyParser::do_nothing);
  add_key("minimum pixel count", 
    KeyArgument::NONE,	&KeyParser::do_nothing);

  add_key("image scaling factor", 
    KeyArgument::LIST_OF_DOUBLES, &image_scaling_factors);
  // KT 07/10/2002 new
  // support for Louvain la Neuve's extension of 3.3
  add_key("quantification units",
    KeyArgument::DOUBLE, &lln_quantification_units);

  add_key("data offset in bytes", 
    KeyArgument::ULONG,	&data_offset);
  add_key("END OF INTERFILE", 
    KeyArgument::NONE,	&KeyParser::stop_parsing);
}


// MJ 17/05/2000 made bool
bool InterfileHeader::post_processing()
{
  if (patient_orientation_index<0 || patient_rotation_index<0)
    return true;
  // warning: relies on index taking same values as enums in PatientPosition
  patient_position.set_rotation(static_cast<PatientPosition::RotationValues>(patient_rotation_index));
  patient_position.set_orientation(static_cast<PatientPosition::OrientationValues>(patient_orientation_index));

  if (number_format_index<0 || 
      static_cast<ASCIIlist_type::size_type>(number_format_index)>=number_format_values.size())
  {
    warning("Interfile internal error: 'number_format_index' out of range\n");
    return true;
  }
  // KT 07/10/2002 new
  // check if bytes_per_pixel is set if the data type is not 'bit'
  if (number_format_index!=0 && bytes_per_pixel<=0)
  {
    warning("Interfile error: 'number of bytes per pixel' keyword should be set\n to a number > 0");
    return true;
  }

  type_of_numbers = NumericType(number_format_values[number_format_index], bytes_per_pixel);
  
  file_byte_order = byte_order_index==0 ? 
    ByteOrder::little_endian : ByteOrder::big_endian;
  
  if(type_of_data_values[type_of_data_index] != "PET")
    warning("Interfile Warning: only 'type of data := PET' supported.\n");

  // KT 07/10/2002 more extensive error checking for matrix_size keyword
  if (matrix_size.size()==0)
  {
    warning("Interfile error: no matrix size keywords present\n");
    return true;
  }
  if (matrix_size[matrix_size.size()-1].size()!=1)
  {
    warning("Interfile error: last dimension (%d) of 'matrix size' cannot be a list of numbers\n",
      matrix_size[matrix_size.size()-1].size());
    return true;
  }
  for (unsigned int dim=0; dim != matrix_size.size(); ++dim)
  {
    if (matrix_size[dim].size()==0)
    {
      warning("Interfile error: dimension (%d) of 'matrix size' not present\n", dim);
      return true;
    }
    for (unsigned int i=0; i != matrix_size[dim].size(); ++i)
    {
      if (matrix_size[dim][i]<=0)
      {
        warning("Interfile error: dimension (%d) of 'matrix size' has a number <= 0 at position\n", dim, i);
        return true;
      }
    }
  }

  for (int frame=0; frame<num_time_frames; frame++)
  {
    if (image_scaling_factors[frame].size() == 1)
    {
      // use the only value for every scaling factor
      image_scaling_factors[frame].resize(matrix_size[matrix_size.size()-1][0]);
      for (unsigned int i=1; i<image_scaling_factors[frame].size(); i++)
	image_scaling_factors[frame][i] = image_scaling_factors[frame][0];
    } 
    else if (static_cast<int>(image_scaling_factors[frame].size()) !=  matrix_size[matrix_size.size()-1][0])
    {
      warning("Interfile error: wrong number of image scaling factors\n");
      return true;
    }
  }
  
  // KT 07/10/2002 new
  // support for non-standard key
  // TODO as there's currently no way to find out if a key was used in the header, we just rely on the
  // fact that the default didn't change. This isn't good enough, but it has to do for now.
  if (lln_quantification_units!=1.)
  {
     const bool all_one = image_scaling_factors[0][0] == 1.;
    for (int frame=0; frame<num_time_frames; frame++)
      for (unsigned int i=0; i<image_scaling_factors[frame].size(); i++)
      {
        // check if all image_scaling_factors are equal to 1 (i.e. the image_scaling_factors keyword 
        // probably never occured) or lln_quantification_units
        if ((all_one && image_scaling_factors[frame][i] != 1.) ||
            (!all_one && image_scaling_factors[frame][i] != lln_quantification_units))
          {
            warning("Interfile error: key 'quantification units' can only be used when either "
                    "image_scaling_factors[] keywords are not present, or have identical values.\n");
            return true;
          }
        // if they're all 1, we set the value to lln_quantification_units
        if (all_one)
          image_scaling_factors[frame][i] = lln_quantification_units;
      }
    if (all_one)
    {
       warning("Interfile warning: non-standard key 'quantification_units' used to set 'image_scaling_factors' to %g\n",
               lln_quantification_units);
    }      
  } // lln_quantification_units
  
  return false;

}

void InterfileHeader::read_matrix_info()
{
  set_variable();

  matrix_labels.resize(num_dimensions);
  matrix_size.resize(num_dimensions);
  pixel_sizes.resize(num_dimensions, 1.);
  
}

void InterfileHeader::read_frames_info()
{
  set_variable();
  image_scaling_factors.resize(num_time_frames);
  for (int i=0; i<num_time_frames; i++)
    image_scaling_factors[i].resize(1, 1.);
  data_offset.resize(num_time_frames, 0UL);
  
}

/***********************************************************************/

// MJ 17/05/2000 made bool
bool InterfileImageHeader::post_processing()
{

  if (InterfileHeader::post_processing() == true)
    return true;

  if (PET_data_type_values[PET_data_type_index] != "Image")
    { warning("Interfile error: expecting an image\n");  return true; }
  
  if (num_dimensions != 3)
    { warning("Interfile error: expecting 3D image\n"); return true; }

  if ( (matrix_size[0].size() != 1) || 
       (matrix_size[1].size() != 1) ||
       (matrix_size[2].size() != 1) )
  { warning("Interfile error: only handling image with homogeneous dimensions\n"); return true; }

  // KT 09/10/98 changed order z,y,x->x,y,z
  // KT 09/10/98 allow no labels at all
  if (matrix_labels[0].length()>0 
      && (matrix_labels[0]!="x" || matrix_labels[1]!="y" ||
	  matrix_labels[2]!="z"))
    {
      warning("Interfile: only supporting x,y,z order of coordinates now.\n");
      return true; 
    }


  return false;
}
/**********************************************************************/

//KT 26/10/98
// KT 13/11/98 moved stream arg from constructor to parse()
InterfilePDFSHeader::InterfilePDFSHeader()
     : InterfileHeader()
{
  num_segments = -1;

  add_key("minimum ring difference per segment",
    KeyArgument::LIST_OF_INTS, 
    (KeywordProcessor)&InterfilePDFSHeader::resize_segments_and_set, 
    &min_ring_difference);
  add_key("maximum ring difference per segment",
    KeyArgument::LIST_OF_INTS, 
    (KeywordProcessor)&InterfilePDFSHeader::resize_segments_and_set, 
    &max_ring_difference);
  
  
  // warning these keys should match what is in Scanner::parameter_info()
  // TODO get Scanner to parse these
  add_key("Scanner parameters",
	  KeyArgument::NONE,	&KeyParser::do_nothing);
  // this is currently ignored (use "originating system" instead)
  add_key("Scanner type",
	  KeyArgument::NONE,	&KeyParser::do_nothing);

  // first set to some crazy values
  num_rings = -1;
  add_key("number of rings", 
	  &num_rings);
  num_detectors_per_ring = -1;
  add_key("number of detectors per ring", 
	  &num_detectors_per_ring);
  transaxial_FOV_diameter_in_cm = -1;
  add_key("transaxial FOV diameter (cm)",
	  &transaxial_FOV_diameter_in_cm);
  inner_ring_diameter_in_cm = -1;
  add_key("inner ring diameter (cm)",
	   &inner_ring_diameter_in_cm);
  average_depth_of_interaction_in_cm = -1;
  add_key("average depth of interaction (cm)",
	  &average_depth_of_interaction_in_cm);
  distance_between_rings_in_cm = -1;
  add_key("distance between rings (cm)",
	  &distance_between_rings_in_cm);
  default_bin_size_in_cm = -1;
  add_key("default bin size (cm)",
	  &default_bin_size_in_cm);
  // this is a good default value
  view_offset_in_degrees = 0;
  add_key("view offset (degrees)",
	  &view_offset_in_degrees);
  max_num_non_arccorrected_bins=0;
  default_num_arccorrected_bins=0;
  add_key("Maximum number of non-arc-corrected bins",
	  &max_num_non_arccorrected_bins);
  add_key("Default number of arc-corrected bins",
	  &default_num_arccorrected_bins);

  num_axial_blocks_per_bucket = 0;
  add_key("number of blocks_per_bucket in axial direction",
	  &num_axial_blocks_per_bucket);
  num_transaxial_blocks_per_bucket = 0;
  add_key("number of blocks_per_bucket in transaxial direction",
	  &num_transaxial_blocks_per_bucket);
  num_axial_crystals_per_block = 0;
  add_key("number of crystals_per_block in axial direction",
	  &num_axial_crystals_per_block);
  num_transaxial_crystals_per_block = 0;
  add_key("number of crystals_per_block in transaxial direction",
	  &num_transaxial_crystals_per_block);
  num_axial_crystals_per_singles_unit = -1;
  add_key("number of crystals_per_singles_unit in axial direction",
	  &num_axial_crystals_per_singles_unit);
  num_transaxial_crystals_per_singles_unit = -1;
  add_key("number of crystals_per_singles_unit in transaxial direction",
	  &num_transaxial_crystals_per_singles_unit);
  // sensible default
  num_detector_layers = 1;
  add_key("number of detector layers",
	  &num_detector_layers);

  add_key("end scanner parameters",
	  KeyArgument::NONE,	&KeyParser::do_nothing);
  
  effective_central_bin_size_in_cm = -1;
  add_key("effective central bin size (cm)",
	  &effective_central_bin_size_in_cm);
  add_key("applied corrections",
    KeyArgument::LIST_OF_ASCII, &applied_corrections);

}

void InterfilePDFSHeader::resize_segments_and_set()
{
  // find_storage_order returns true if already found (or error)
  if (num_segments < 0 && !find_storage_order())
  {
    min_ring_difference.resize(num_segments);
    max_ring_difference.resize(num_segments);
    
  }
  
  if (num_segments >= 0)
    set_variable();
  
}

int InterfilePDFSHeader::find_storage_order()
{
  
  if (num_dimensions != 4)
  { 
    warning("Interfile error: expecting 4D structure "); 
    stop_parsing();
    return true; 
  }
  
  if (matrix_labels[0] != "tangential coordinate")
  { 
    // use error message with index [1] as that is what the user sees.
    warning("Interfile error: expecting 'matrix axis label[1] := tangential coordinate'\n"); 
    stop_parsing();
    return true; 
  }
  num_bins = matrix_size[0][0];
  
  if (matrix_labels[3] == "segment")
  {
    num_segments = matrix_size[3][0];
    
    if (matrix_labels[1] == "axial coordinate" && matrix_labels[2] == "view")
    {
      storage_order =ProjDataFromStream::Segment_View_AxialPos_TangPos;
      num_views = matrix_size[2][0];
#ifdef _MSC_VER
      num_rings_per_segment.assign(matrix_size[1].begin(), matrix_size[1].end());
#else      
      num_rings_per_segment = matrix_size[1];
#endif
    }
    else if (matrix_labels[1] == "view" && matrix_labels[2] == "axial coordinate")
    {
      storage_order = ProjDataFromStream::Segment_AxialPos_View_TangPos;
      num_views = matrix_size[1][0];
#ifdef _MSC_VER
      
      num_rings_per_segment.assign(matrix_size[2].begin(), matrix_size[2].end());
      
#else
      num_rings_per_segment = matrix_size[2];
#endif
    }
    
  }
  /*
  else if (matrix_labels[3] == "view" && 
  matrix_labels[2] == "segment" && matrix_labels[1] == "axial coordinate")
  {
  storage_order = ProjDataFromStream::View_Segment_AxialPos_TangPos;
  num_segments = matrix_size[2][0];
  num_views = matrix_size[3][0];
  #ifdef _MSC_VER
  num_rings_per_segment.assign(matrix_size[1].begin(), matrix_size[1].end());
  #else
  num_rings_per_segment = matrix_size[1];
  #endif
  
   }
  */
  else
  { 
    warning("Interfile error: matrix labels not in expected (or supported) format\n"); 
    stop_parsing();
    return true; 
  }
  
  return false;
  
}

// definition for using sort() below.
// This is a function object that allows comparing the first elements of 2 
// pairs.
template <class T1, class T2>
class compare_first :
public binary_function<T1, T1, bool> 
{
public:
  bool operator()(const pair<T1, T2>& p1, const pair<T1, T2>& p2)  const
  {
    return p1.first < p2.first;
  }
};


// This function assigns segment numbers by sorting the average 
// ring differences. It returns a list of the segment numbers 
// in the same order as the min/max_ring_difference vectors
void
find_segment_sequence(vector<int>& segment_sequence,
                      VectorWithOffset<int>& sorted_num_rings_per_segment,
		      VectorWithOffset<int>& sorted_min_ring_diff,
		      VectorWithOffset<int>& sorted_max_ring_diff,
		      vector<int>& num_rings_per_segment,
		      const vector<int>& min_ring_difference, 
		      const vector<int>& max_ring_difference)
{
  const int num_segments = min_ring_difference.size();
  assert(num_segments%2 == 1);
  
  
  vector< pair<float, int> > sum_and_location(num_segments);
  for (int i=0; i<num_segments; i++)
  {
    sum_and_location[i].first = static_cast<float>(min_ring_difference[i] + max_ring_difference[i]);
    sum_and_location[i].second = i;
  }
#if 0
  cerr<< "DISPLAY SUM and LOCATION\n"<<endl;
  
  cerr<<"SUM\n"<<endl;
  for(unsigned int i = 0;i<sum_and_location.size();i++)
  {
    cerr<< sum_and_location[i].first<<" ";
  }
  cerr<<endl;
  
  cerr<<"Location\n"<<endl;
  for(unsigned int i = 0;i<sum_and_location.size();i++)
  {
    cerr<< sum_and_location[i].second<<" ";
  }
  cerr<<endl;
#endif  
  
  // sort with respect to 'sum'
  std::sort(sum_and_location.begin(), sum_and_location.end(),  
    compare_first<float, int>());
#if 0  
  cerr<<"display  sum_sorted"<<endl;
  for(unsigned int i = 0;i<sum_and_location.size();i++)
  {
    cerr<< sum_and_location[i].first<<" ";
  }
  cerr<<endl;
#endif  
  
  
  // find number of segment 0
  int segment_zero_num = 0;
  while (segment_zero_num < num_segments &&
    sum_and_location[segment_zero_num].first < -1E-3)
    segment_zero_num++;
  
  if (segment_zero_num == num_segments ||
    sum_and_location[segment_zero_num].first > 1E-3)
  {
  error("This data does not seem to contain segment 0. \n"
    "We can't handle this at the moment. Sorry.");
  }
  
  vector< pair<int, int> > location_and_segment_num(num_segments);
  for (int i=0; i<num_segments; i++)
  {
    location_and_segment_num[i].first = sum_and_location[i].second;
    location_and_segment_num[i].second = i - segment_zero_num;
  }

#if 0
  cerr<< "display location segment\n"<<endl;
  for(unsigned int i = 0;i<location_and_segment_num.size();i++)
  {
    cerr<< location_and_segment_num[i].first<<" ";
  }
  cerr<<endl;
  
  cerr<< "display segment\n"<<endl;
  for(unsigned int i = 0;i<location_and_segment_num.size();i++)
  {
    cerr<< location_and_segment_num[i].second<<" ";
  }
  cerr<<endl;
#endif
  
  const int min_segment_num = location_and_segment_num[0].second;
  const int max_segment_num = location_and_segment_num[num_segments-1].second;

  // KT 19/05/2000 replaced limit with min/max_segment_num
  //int limit = static_cast<int>(ceil(num_segments/2 ));
  
  sorted_min_ring_diff = VectorWithOffset<int>(min_segment_num,max_segment_num);
  sorted_max_ring_diff = VectorWithOffset<int>(min_segment_num,max_segment_num);
  sorted_num_rings_per_segment= VectorWithOffset<int>(min_segment_num,max_segment_num);
  
  
  for (int i=0; i<num_segments; i++)
  {
    sorted_min_ring_diff[(location_and_segment_num[i].second)]
      = min_ring_difference[(location_and_segment_num[i].first)];
    
    sorted_max_ring_diff[(location_and_segment_num[i].second)]
      = max_ring_difference[(location_and_segment_num[i].first)];
    
    sorted_num_rings_per_segment[(location_and_segment_num[i].second)]
      = num_rings_per_segment[(location_and_segment_num[i].first)];
    
    
  }

#if 0
  cerr<< "sorted_min_ring_diff\n"<<endl;
  for( int i =min_segment_num;i<max_segment_num;i++)
  {
    cerr<< sorted_min_ring_diff[i]<<" ";
  }

  cerr<<endl;

  cerr<< "sorted_max_ring_diff\n"<<endl;
  for( int i =min_segment_num;i<max_segment_num;i++)
  {
    cerr<< sorted_max_ring_diff[i]<<" ";
  }
  cerr<<endl;

  
  cerr<< "sorted_num_rings_per_segment\n"<<endl;
  for( int i =min_segment_num;i<max_segment_num;i++)
  {
    cerr<< sorted_num_rings_per_segment[i]<<" ";
  }
  cerr<<endl;
#endif

 
  // sort back to original location
  sort(location_and_segment_num.begin(), location_and_segment_num.end(),  
      compare_first<int, int>());
   

   segment_sequence.resize(num_segments);  
    for (int i=0; i<num_segments; i++)
      segment_sequence[i] = location_and_segment_num[i].second;
  
#if 0    
  cerr<< "segment sequence\n"<<endl;
  for(unsigned int i =0;i<segment_sequence.size();i++)
  {
    cerr<< segment_sequence[i]<<" ";
  }
  cerr<<endl;
#endif

    
  //}
	 
	  
}	  
	  
// MJ 17/05/2000 made bool
bool InterfilePDFSHeader::post_processing()
{
  
  if (InterfileHeader::post_processing() == true)
    return true;
  
  if (PET_data_type_values[PET_data_type_index] != "Emission")
  { warning("Interfile error: expecting emission data\n");  return true; }
  
  if (min_ring_difference.size()!= static_cast<unsigned int>(num_segments))
  { 
    warning("Interfile error: per-segment information is inconsistent: min_ring_difference\n"); 
    return true;
  }
  if (max_ring_difference.size() != static_cast<unsigned int>(num_segments))
  { 
    warning("Interfile error: per-segment information is inconsistent: max_ring_difference\n"); 
    return true;
  }
  if (num_rings_per_segment.size()!= static_cast<unsigned int>(num_segments))
  { 
    warning("Interfile error: per-segment information is inconsistent: num_rings_per_segment\n"); 
    return true;
  }
  
  // check for arc-correction
  if (applied_corrections.size() == 0)
  {
    warning("\nParsing Interfile header for projection data: \n"
            "\t'applied corrections' keyword not found. Assuming arc-corrected data\n");
    is_arccorrected = true;
  }
  else
  {
    is_arccorrected = false;
    for (
#ifndef STIR_NO_NAMESPACES
      std::
#endif
      vector<string>::const_iterator iter = applied_corrections.begin();
         iter != applied_corrections.end();
         ++iter)
    {
      const string correction = standardise_keyword(*iter);
      if(correction == "arc correction" || correction == "arc corrected")
      {
        is_arccorrected = true;
        break;
      }
      else if (correction != "none")
	    warning("\nParsing Interfile header for projection data: \n"
		    "\t value '%s' for keyword 'applied corrections' ignored\n",
		    correction.c_str());
	
    }
    
  }
 
  VectorWithOffset<int> sorted_min_ring_diff;
  VectorWithOffset<int> sorted_max_ring_diff;
  VectorWithOffset<int> sorted_num_rings_per_segment;
  
  find_segment_sequence( segment_sequence,sorted_num_rings_per_segment,
    sorted_min_ring_diff,sorted_max_ring_diff,
    num_rings_per_segment,
    min_ring_difference, max_ring_difference);
#if 0  
  cerr << "PDFS data read inferred header :\n";
  cerr << "Segment sequence :";
  for (unsigned int i=0; i<segment_sequence.size(); i++)
    cerr << segment_sequence[i] << "  ";
  cerr << endl;
  cerr << "RingDiff minimum :";
  for (int i=sorted_min_ring_diff.get_min_index(); i<=sorted_min_ring_diff.get_max_index(); i++)
    cerr <<sorted_min_ring_diff[i] << "  ";  cerr << endl;
  cerr << "RingDiff maximum :";
  for (int i=sorted_max_ring_diff.get_min_index(); i<=sorted_max_ring_diff.get_max_index(); i++)
    cerr << sorted_max_ring_diff[i] << "  ";  cerr << endl;
  cerr << "Nbplanes/segment :";
  for (int i=sorted_num_rings_per_segment.get_min_index(); i<=sorted_num_rings_per_segment.get_max_index(); i++)
    cerr << sorted_num_rings_per_segment[i] << "  ";  cerr << endl;

  cerr << "Total number of planes :" 
    << 
#ifndef STIR_NO_NAMESPACES // stupid work-around for VC
    std::accumulate
#else
    accumulate
#endif
       (num_rings_per_segment.begin(), num_rings_per_segment.end(), 0)
    << endl;
#endif
  
  // handle scanner

  shared_ptr<Scanner> guessed_scanner_ptr = 
    Scanner::get_scanner_from_name(originating_system);
  bool originating_system_was_recognised = 
    guessed_scanner_ptr->get_type() != Scanner::Unknown_scanner;
  if (!originating_system_was_recognised)
  {
    // feable attempt to guess the system by checking the num_views etc

    char * warning_msg = 0;
    if (num_detectors_per_ring < 1)
    {
      num_detectors_per_ring = num_views*2;
      warning_msg = "\nInterfile warning: I don't recognise 'originating system' value.\n"
	"\tI guessed %s from 'num_views' (note: this guess is wrong for mashed data)\n"
	" and 'number of rings'\n";
    }
    else
    {
      warning_msg = "\nInterfile warning: I don't recognise 'originating system' value.\n"
	"I guessed %s from 'number of detectors per ring' and 'number of rings'\n";
    }
    
    
    switch (num_detectors_per_ring)
    {
    case 192*2:
      guessed_scanner_ptr = new Scanner( Scanner::E953);
      warning(warning_msg, "ECAT 953");
      break;
    case 336*2:
      guessed_scanner_ptr = new Scanner( Scanner::Advance);
      warning(warning_msg, "Advance");
      break;
    case 288*2:
      if(num_rings == 104) 
      { //added by Dylan Togane
 	guessed_scanner_ptr = new Scanner( Scanner::HRRT);
	warning(warning_msg, "HRRT");
      }
      else if (num_rings == 48)
      {
	guessed_scanner_ptr = new Scanner( Scanner::E966);
	warning(warning_msg, "ECAT 966");
      }
      else if (num_rings == 32)
      {
	guessed_scanner_ptr = new Scanner( Scanner::E962);
	warning(warning_msg, "ECAT 962");
      }
      break; // Dylan Togane [dtogane@camhpet.on.ca] 30/07/2002 bug fix: added break
    case 256*2:
      guessed_scanner_ptr = new Scanner( Scanner::E951);
      warning(warning_msg, "ECAT 951");
      break;
    }

    if (guessed_scanner_ptr->get_type() == Scanner::Unknown_scanner)
      warning("\nInterfile warning: I did not recognise the scanner neither from \n"
	      "'originating_system' or 'number of detectors per ring' and 'number of rings'.\n");    
  }

  bool mismatch_between_header_and_guess = false;
 
  if (guessed_scanner_ptr->get_type() != Scanner::Unknown_scanner &&
      guessed_scanner_ptr->get_type() != Scanner::User_defined_scanner)
  {
     // fill in values which are not in the Interfile header
    
    if (num_rings < 1)
      num_rings = guessed_scanner_ptr->get_num_rings();
    if (num_detectors_per_ring < 1)
      num_detectors_per_ring = guessed_scanner_ptr->get_max_num_views()*2;
#if 0
    if (transaxial_FOV_diameter_in_cm < 0)
      transaxial_FOV_diameter_in_cm = guessed_scanner_ptr->FOV_radius*2/10.;
#endif
    if (inner_ring_diameter_in_cm < 0)
      inner_ring_diameter_in_cm = guessed_scanner_ptr->get_inner_ring_radius()*2/10.;
    if (average_depth_of_interaction_in_cm < 0)
      average_depth_of_interaction_in_cm = guessed_scanner_ptr->get_average_depth_of_interaction()/10;
    if (distance_between_rings_in_cm < 0)
      distance_between_rings_in_cm = guessed_scanner_ptr->get_ring_spacing()/10;
    if (default_bin_size_in_cm < 0)
      default_bin_size_in_cm = 
         guessed_scanner_ptr->get_default_bin_size()/10;
    if (max_num_non_arccorrected_bins <= 0)
      max_num_non_arccorrected_bins = guessed_scanner_ptr->get_max_num_non_arccorrected_bins();
    if (default_num_arccorrected_bins <= 0)
      default_num_arccorrected_bins = guessed_scanner_ptr->get_default_num_arccorrected_bins();


    if (num_axial_blocks_per_bucket<=0)
      num_axial_blocks_per_bucket = guessed_scanner_ptr->get_num_axial_blocks_per_bucket();
    if (num_transaxial_blocks_per_bucket<=0)
      num_transaxial_blocks_per_bucket = guessed_scanner_ptr->get_num_transaxial_blocks_per_bucket();
    if (num_axial_crystals_per_block<=0)
      num_axial_crystals_per_block = guessed_scanner_ptr->get_num_axial_crystals_per_block();
    if (num_transaxial_crystals_per_block<=0)
      num_transaxial_crystals_per_block = guessed_scanner_ptr->get_num_transaxial_crystals_per_block();
    if (num_axial_crystals_per_singles_unit < 0) 
      num_axial_crystals_per_singles_unit = 
        guessed_scanner_ptr->get_num_axial_crystals_per_singles_unit();
    if (num_transaxial_crystals_per_singles_unit < 0) 
      num_transaxial_crystals_per_singles_unit = 
        guessed_scanner_ptr->get_num_transaxial_crystals_per_singles_unit();
    if (num_detector_layers<=0)
      num_detector_layers = guessed_scanner_ptr->get_num_detector_layers();
    
    // consistency check with values of the guessed_scanner_ptr we guessed above

    if (num_rings != guessed_scanner_ptr->get_num_rings())
      {
	warning("Interfile warning: 'number of rings' (%d) is expected to be %d.\n",
		num_rings, guessed_scanner_ptr->get_num_rings());
	mismatch_between_header_and_guess = true;
      }
    if (num_detectors_per_ring != guessed_scanner_ptr->get_num_detectors_per_ring())
      {
	warning("Interfile warning: 'number of detectors per ring' (%d) is expected to be %d.\n",
		num_detectors_per_ring, guessed_scanner_ptr->get_num_detectors_per_ring());
	mismatch_between_header_and_guess = true;
      }
    if (fabs(inner_ring_diameter_in_cm - guessed_scanner_ptr->get_inner_ring_radius()*2/10.) > .001)
      {
	warning("Interfile warning: 'inner ring diameter (cm)' (%f) is expected to be %f.\n",
		inner_ring_diameter_in_cm, guessed_scanner_ptr->get_inner_ring_radius()*2/10.);
	mismatch_between_header_and_guess = true;
      }
    if (fabs(average_depth_of_interaction_in_cm - 
             guessed_scanner_ptr->get_average_depth_of_interaction()/10) > .001)
      {
	warning("Interfile warning: 'average depth of interaction (cm)' (%f) is expected to be %f.\n",
		average_depth_of_interaction_in_cm, 
                guessed_scanner_ptr->get_average_depth_of_interaction()/10);
	mismatch_between_header_and_guess = true;
      }
    if (fabs(distance_between_rings_in_cm-guessed_scanner_ptr->get_ring_spacing()/10) > .001)
      {
	warning("Interfile warning: 'distance between rings (cm)' (%f) is expected to be %f.\n",
		distance_between_rings_in_cm, guessed_scanner_ptr->get_ring_spacing()/10);
	mismatch_between_header_and_guess = true;
      }
    if (fabs(default_bin_size_in_cm-guessed_scanner_ptr->get_default_bin_size()/10) > .001)
      {
	warning("Interfile warning: 'default bin size (cm)' (%f) is expected to be %f.\n",
		default_bin_size_in_cm, guessed_scanner_ptr->get_default_bin_size()/10);
	mismatch_between_header_and_guess = true;
      }
    if (max_num_non_arccorrected_bins - guessed_scanner_ptr->get_max_num_non_arccorrected_bins())
      {
	warning("Interfile warning: 'max_num_non_arccorrected_bins' (%d) is expected to be %d",
		max_num_non_arccorrected_bins, guessed_scanner_ptr->get_max_num_non_arccorrected_bins());
	mismatch_between_header_and_guess = true;
      }
    if (default_num_arccorrected_bins - guessed_scanner_ptr->get_default_num_arccorrected_bins())
      {
	warning("Interfile warning: 'default_num_arccorrected_bins' (%d) is expected to be %d",
		default_num_arccorrected_bins, guessed_scanner_ptr->get_default_num_arccorrected_bins());
	mismatch_between_header_and_guess = true;
      }
    if (
	guessed_scanner_ptr->get_num_transaxial_blocks_per_bucket()>0 &&
	num_transaxial_blocks_per_bucket != guessed_scanner_ptr->get_num_transaxial_blocks_per_bucket())
      {
	warning("Interfile warning: num_transaxial_blocks_per_bucket (%d) is expected to be %d.\n",
		num_transaxial_blocks_per_bucket, guessed_scanner_ptr->get_num_transaxial_blocks_per_bucket());
	mismatch_between_header_and_guess = true;
      }
    if (
	guessed_scanner_ptr->get_num_axial_blocks_per_bucket()>0 &&
	num_axial_blocks_per_bucket != guessed_scanner_ptr->get_num_axial_blocks_per_bucket())
      {
	warning("Interfile warning: num_axial_blocks_per_bucket (%d) is expected to be %d.\n",
		num_axial_blocks_per_bucket, guessed_scanner_ptr->get_num_axial_blocks_per_bucket());
	mismatch_between_header_and_guess = true;
      }
    if (
	guessed_scanner_ptr->get_num_axial_crystals_per_block()>0 &&
	num_axial_crystals_per_block!= guessed_scanner_ptr->get_num_axial_crystals_per_block())
      {
	warning("Interfile warning: num_axial_crystals_per_block (%d) is expected to be %d.\n",
		num_axial_crystals_per_block, guessed_scanner_ptr->get_num_axial_crystals_per_block());
      	mismatch_between_header_and_guess = true;
      }
    if (
	guessed_scanner_ptr->get_num_transaxial_crystals_per_block()>0 &&
	num_transaxial_crystals_per_block!= guessed_scanner_ptr->get_num_transaxial_crystals_per_block())
      {
	warning("Interfile warning: num_transaxial_crystals_per_block (%d) is expected to be %d.\n",
		num_transaxial_crystals_per_block, guessed_scanner_ptr->get_num_transaxial_crystals_per_block());
	mismatch_between_header_and_guess = true;
      }
    if ( guessed_scanner_ptr->get_num_axial_crystals_per_singles_unit() > 0 &&
         num_axial_crystals_per_singles_unit != 
         guessed_scanner_ptr->get_num_axial_crystals_per_singles_unit() ) 
      {
        warning("Interfile warning: axial crystals per singles unit (%d) is expected to be %d.\n",
		num_axial_crystals_per_singles_unit, 
                guessed_scanner_ptr->get_num_axial_crystals_per_singles_unit());
	mismatch_between_header_and_guess = true;
      }
    if ( guessed_scanner_ptr->get_num_transaxial_crystals_per_singles_unit() > 0 &&
         num_transaxial_crystals_per_singles_unit != 
         guessed_scanner_ptr->get_num_transaxial_crystals_per_singles_unit() ) 
      {
        warning("Interfile warning: transaxial crystals per singles unit (%d) is expected to be %d.\n",
		num_transaxial_crystals_per_singles_unit, 
                guessed_scanner_ptr->get_num_transaxial_crystals_per_singles_unit());
	mismatch_between_header_and_guess = true;
      }
    if (
	guessed_scanner_ptr->get_num_detector_layers()>0 &&
	num_detector_layers != guessed_scanner_ptr->get_num_detector_layers())
      {
	warning("Interfile warning: num_detector_layers (%d) is expected to be %d.\n",
		num_detector_layers, guessed_scanner_ptr->get_num_detector_layers());
	mismatch_between_header_and_guess = true;
      }

    // end of checks. If they failed, we ignore the guess
    if (mismatch_between_header_and_guess)
      {
	warning("Interfile warning: I have used all explicit settings for the scanner\n"
		"\tfrom the Interfile header, and remaining fields set from the\n"
		"\t%s model.\n",
		guessed_scanner_ptr->get_name().c_str());
	if (!originating_system_was_recognised)
	  guessed_scanner_ptr = new Scanner( Scanner::Unknown_scanner);
      }
    }

  if (guessed_scanner_ptr->get_type() == Scanner::Unknown_scanner ||
      guessed_scanner_ptr->get_type() == Scanner::User_defined_scanner)
  {
    // warn if the Interfile header does not provide enough info

    if (num_rings < 1)
      warning("Interfile warning: 'number of rings' invalid.\n");
    if (num_detectors_per_ring < 1)
      warning("Interfile warning: 'num_detectors_per_ring' invalid.\n");
#if 0
    if (transaxial_FOV_diameter_in_cm < 0)
      warning("Interfile warning: 'transaxial FOV diameter (cm)' invalid.\n");
#endif
    if (inner_ring_diameter_in_cm <= 0)
      warning("Interfile warning: 'inner ring diameter (cm)' invalid. This might disastrous\n");
    if (average_depth_of_interaction_in_cm <= 0)
      warning("Interfile warning: 'average depth of interaction (cm)' invalid. This might be disastrous.\n");
    if (distance_between_rings_in_cm <= 0)
      warning("Interfile warning: 'distance between rings (cm)' invalid.\n");
    if (default_bin_size_in_cm <= 0)
      warning("Interfile warning: 'default_bin size (cm)' invalid.\n");
    if (num_axial_crystals_per_singles_unit <= 0)
      warning("Interfile warning: 'axial crystals per singles unit' invalid.\n");
    if (num_transaxial_crystals_per_singles_unit <= 0)
      warning("Interfile warning: 'transaxial crystals per singles unit' invalid.\n");

  }

  // finally, we construct a new scanner object with
  // data from the Interfile header (or the guessed scanner).
  shared_ptr<Scanner> scanner_ptr_from_file =
    new Scanner(guessed_scanner_ptr->get_type(), 
                originating_system,
		num_detectors_per_ring, 
                num_rings, 
		max_num_non_arccorrected_bins, 
		default_num_arccorrected_bins,
		static_cast<float>(inner_ring_diameter_in_cm*10./2),
                static_cast<float>(average_depth_of_interaction_in_cm/10),
		static_cast<float>(distance_between_rings_in_cm*10.),
		static_cast<float>(default_bin_size_in_cm*10),
		static_cast<float>(view_offset_in_degrees*_PI/180),
		num_axial_blocks_per_bucket, 
		num_transaxial_blocks_per_bucket,
		num_axial_crystals_per_block,
		num_transaxial_crystals_per_block,
		num_axial_crystals_per_singles_unit,
                num_transaxial_crystals_per_singles_unit,
                num_detector_layers);

  bool is_consistent =
    scanner_ptr_from_file->check_consistency() == Succeeded::yes;
  if (scanner_ptr_from_file->get_type() == Scanner::Unknown_scanner ||
      scanner_ptr_from_file->get_type() == Scanner::User_defined_scanner ||
      mismatch_between_header_and_guess ||
      !is_consistent)
    {
      warning("Interfile parsing ended up with the following scanner:\n%s\n",
	      scanner_ptr_from_file->parameter_info().c_str());
    }
 
  
  // float azimuthal_angle_sampling =_PI/num_views;
  
  
   
  
  if (is_arccorrected)
    {
      if (effective_central_bin_size_in_cm <= 0)
	effective_central_bin_size_in_cm =
	  scanner_ptr_from_file->get_default_bin_size()/10;
      else if (fabs(effective_central_bin_size_in_cm - 
		    scanner_ptr_from_file->get_default_bin_size()/10)>.001)	
	warning("Interfile warning: unexpected effective_central_bin_size_in_cm\n"
		"Value in header is %g while the default for the scanner is %g\n"
		"Using value from header.",
		effective_central_bin_size_in_cm,
		scanner_ptr_from_file->get_default_bin_size()/10);
      
      data_info_ptr = 
	new ProjDataInfoCylindricalArcCorr (
					    scanner_ptr_from_file,
					    float(effective_central_bin_size_in_cm*10.),
					    sorted_num_rings_per_segment,
					    sorted_min_ring_diff,
					    sorted_max_ring_diff,
					    num_views,num_bins);
    }
  else
    {
      data_info_ptr = 
	new ProjDataInfoCylindricalNoArcCorr (
					      scanner_ptr_from_file,
					      sorted_num_rings_per_segment,
					      sorted_min_ring_diff,
					      sorted_max_ring_diff,
					      num_views,num_bins);
      if (effective_central_bin_size_in_cm>0 &&
	  fabs(effective_central_bin_size_in_cm - 
	       data_info_ptr->get_sampling_in_s(Bin(0,0,0,0))/10.)>.01)
	{
	  warning("Interfile warning: inconsistent effective_central_bin_size_in_cm\n"
		  "Value in header is %g while I expect %g from the inner ring radius etc\n"
		  "Ignoring value in header",
		  effective_central_bin_size_in_cm,
		  data_info_ptr->get_sampling_in_s(Bin(0,0,0,0))/10.);
	}
    }
  //cerr << data_info_ptr->parameter_info() << endl;
  
  return false;
}

END_NAMESPACE_STIR
