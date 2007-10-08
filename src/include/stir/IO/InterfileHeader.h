//
// $Id$
//
/*
    Copyright (C) 2002- $Date$, Hammersmith Imanet Ltd
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
  \brief  This file declares the classes stir::InterfileHeader,
          stir::InterfileImageHeader, stir::InterfilePDFSHeader  

  \author Kris Thielemans
  \author Sanida Mustafovic
  \author PARAPET project

  $Date$
  $Revision$

  See http://www.HammersmithImanet.com/~kris for a description of the full
  proposal for Interfile headers for 3D PET.
*/


#ifndef __stir_INTERFILEHEADER_H__
#define __stir_INTERFILEHEADER_H__

#include "stir/ByteOrder.h"
#include "stir/NumericInfo.h"
#include "stir/KeyParser.h"
#include "stir/PatientPosition.h"
#include "stir/ProjDataFromStream.h"
#include "stir/TimeFrameDefinitions.h"


START_NAMESPACE_STIR

class ProjDataInfo;

/*!
  \brief a class for Interfile keywords (and parsing) common to 
  all types of data
  \ingroup InterfileIO
  */
class InterfileHeader : public KeyParser
{
public:
  //! A value that can be used to signify that a variable has not been set during parsing.
  static const double double_value_not_set;

  InterfileHeader();

  virtual ~InterfileHeader() {}

protected:
  // Returns false if OK, true if not.
  virtual bool post_processing();


private:

  // TODO the next few ones should be made static members
  // Lists of possible values for some keywords
  ASCIIlist_type number_format_values;	
  ASCIIlist_type byte_order_values;
  ASCIIlist_type type_of_data_values;
  ASCIIlist_type patient_orientation_values;
  ASCIIlist_type patient_rotation_values;

  // Corresponding variables here

  int number_format_index;
  int byte_order_index;
  int type_of_data_index;
  int patient_orientation_index;
  int patient_rotation_index;

  // Extra private variables which will be translated to something more useful
  int bytes_per_pixel;

  // Louvain la Neuve style of 'image scaling factors'
  double lln_quantification_units;
 protected:
  virtual void read_matrix_info();
  void read_frames_info();

public :

  string originating_system;
  
  ASCIIlist_type PET_data_type_values;	
  int PET_data_type_index;

  // TODO these shouldn't be here, but in PETStudy or something

  // 'Final' variables

  PatientPosition patient_position;

  string data_file_name;

  //! This will be determined from number_format_index and bytes_per_pixel
  NumericType		type_of_numbers;
  //! This will be determined from byte_order_index, or just keep its default value;
  ByteOrder file_byte_order;
	
  int			num_dimensions;
  int			num_time_frames;
  std::vector<std::string>	matrix_labels;
  std::vector<std::vector<int> > matrix_size; 
  std::vector<double>	pixel_sizes;
  std::vector<double> image_relative_start_times;
  std::vector<double> image_durations;
  std::vector<std::vector<double> > image_scaling_factors;
  std::vector<unsigned long> data_offset;

  TimeFrameDefinitions time_frame_definitions;
};


/*!
  \brief a class for Interfile keywords (and parsing) specific to images
  \ingroup InterfileIO
  */
class InterfileImageHeader : public InterfileHeader
{
 private:
  typedef InterfileHeader base_type;

public:
  InterfileImageHeader();
  std::vector<double>	first_pixel_offsets;

protected:
  virtual void read_matrix_info();
  //! Returns false if OK, true if not.
  virtual bool post_processing();

};

/*!
  \brief a class for Interfile keywords (and parsing) specific to 
  projection data (i.e. ProjDataFromStream)
  \ingroup InterfileIO
  */
class InterfilePDFSHeader : public InterfileHeader
{
public:
  InterfilePDFSHeader();

protected:

  //! Returns false if OK, true if not.
  virtual bool post_processing();

public:
 
  vector<int> segment_sequence;
  vector<int> min_ring_difference; 
  vector<int> max_ring_difference; 
  vector<int> num_rings_per_segment;

  vector<string> applied_corrections;
 
  // derived values
  int num_segments;
  int num_views;
  int num_bins;
  ProjDataFromStream::StorageOrder storage_order;
  ProjDataInfo* data_info_ptr;

private:
  void resize_segments_and_set();
  int find_storage_order();

  // members that will be used to set Scanner
  // TODO parsing should be moved to Scanner
  int num_rings;
  int num_detectors_per_ring;
  
  double transaxial_FOV_diameter_in_cm;
  double inner_ring_diameter_in_cm;
  double average_depth_of_interaction_in_cm;
  double distance_between_rings_in_cm;
  double default_bin_size_in_cm;
  // this intrinsic tilt
  double view_offset_in_degrees;
  int max_num_non_arccorrected_bins;
  int default_num_arccorrected_bins;


  int num_axial_blocks_per_bucket;
  int num_transaxial_blocks_per_bucket;
  int num_axial_crystals_per_block;
  int num_transaxial_crystals_per_block;
  int num_axial_crystals_per_singles_unit;
  int num_transaxial_crystals_per_singles_unit;
  int num_detector_layers;
  // end scanner parameters

  double effective_central_bin_size_in_cm;
  bool is_arccorrected;
};

END_NAMESPACE_STIR

#endif // __stir_INTERFILEHEADER_H__
