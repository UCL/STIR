/*
    Copyright (C) 2002-2007, Hammersmith Imanet Ltd
    Copyright (C) 2013, 2016, 2018 University College London
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
  \author Richard Brown

  See http://stir.sourceforge.net for a description of the full
  proposal for Interfile headers for 3D PET.
*/


#ifndef __stir_INTERFILEHEADER_H__
#define __stir_INTERFILEHEADER_H__

#include "stir/ByteOrder.h"
#include "stir/NumericInfo.h"
#include "stir/KeyParser.h"
#include "stir/ProjDataFromStream.h"
#include "stir/ExamInfo.h"

START_NAMESPACE_STIR

class ProjDataInfo;

/*!
  \brief a minimal class for Interfile keywords (and parsing) common to 
  all types of data

  This class is only used to select which version of Interfile to use.

  \ingroup InterfileIO
  */
class MinimalInterfileHeader : public KeyParser
  {
  public:
    //! A value that can be used to signify that a variable has not been set during parsing.
    static const double double_value_not_set;
    MinimalInterfileHeader();

    virtual ~MinimalInterfileHeader() {}
  protected:
    shared_ptr<ExamInfo> exam_info_sptr;

  private:
    std::string imaging_modality_as_string;
    void set_imaging_modality();
  
  public:
    //! Get a pointer to the exam information
    const ExamInfo*
      get_exam_info_ptr() const;

    //! Get a shared pointer to the exam information
    shared_ptr<ExamInfo>
      get_exam_info_sptr() const;

    std::string version_of_keys;

    std::string siemens_mi_version;
  };

/*!
\brief a class for Interfile keywords (and parsing) common to
all types of data

\ingroup InterfileIO
*/
class InterfileHeader : public MinimalInterfileHeader
{
public:
  InterfileHeader();
  // Returns false if OK, true if not.
  virtual bool post_processing();

private:

  // TODO the next few ones should be made static members
  // Lists of possible values for some keywords
  ASCIIlist_type number_format_values;	
  ASCIIlist_type byte_order_values;
  ASCIIlist_type patient_orientation_values;
  ASCIIlist_type patient_rotation_values;

  // Corresponding variables here

  int number_format_index;
  int byte_order_index;
  int patient_orientation_index;
  int patient_rotation_index;

  void set_type_of_data();
protected:
  int			num_time_frames;
  std::vector<double> image_relative_start_times;
  std::vector<double> image_durations;
  int bytes_per_pixel;
private:

  // Louvain la Neuve style of 'image scaling factors'
  double lln_quantification_units;

  
 protected:
  virtual void read_matrix_info();
  void read_frames_info();
  virtual int get_num_data_types() const { return num_time_frames; }

public :

  ASCIIlist_type type_of_data_values;
  int type_of_data_index;

  ASCIIlist_type PET_data_type_values;	
  int PET_data_type_index;

  ASCIIlist_type process_status_values;
  int process_status_index;

  // 'Final' variables

  std::string data_file_name;

  //! This will be determined from number_format_index and bytes_per_pixel
  NumericType		type_of_numbers;
  //! This will be determined from byte_order_index, or just keep its default value;
  ByteOrder file_byte_order;
	
  int			num_dimensions;
  std::vector<std::string>	matrix_labels;
  std::vector<std::vector<int> > matrix_size; 
  std::vector<double>	pixel_sizes;
  std::vector<std::vector<double> > image_scaling_factors;
  std::vector<unsigned long> data_offset_each_dataset;

  // Acquisition parameters
  //!
  //! \brief lower_en_window_thres
  //! \details Low energy window limit
  float lower_en_window_thres;

  //!
  //! \brief upper_en_window_thres
  //! \details High energy window limit
  float upper_en_window_thres;
  // end acquisition parameters
  
 protected:
  // version 3.3 had only a single offset. we'll internally replace it with data_offset_each_dataset
  unsigned long data_offset;
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
  int num_image_data_types;
  std::vector<std::string> index_nesting_level;
  std::vector<std::string> image_data_type_description;

protected:
  virtual void read_matrix_info();
  //! Returns false if OK, true if not.
  virtual bool post_processing();
  /// Read image data types
  void read_image_data_types();
  //!
  //! \brief Get the number of data types
  //! \details no. time frames * no. data types (kinetic params) * no. gates
  //! Currently, this is only implemented for either multiple time frames OR multiple data types (gates not considered).
  virtual int get_num_data_types() const { return num_time_frames*num_image_data_types; }

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
 
  std::vector<int> segment_sequence;
  std::vector<int> min_ring_difference; 
  std::vector<int> max_ring_difference; 
  std::vector<int> num_rings_per_segment;

  std::vector<std::string> applied_corrections;
  float bed_position_horizontal;
  float bed_position_vertical;
 
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
  //! Energy resolution of the system in keV.
  float energy_resolution;
  //! Reference energy.
  float reference_energy;
  // end scanner parameters

  double effective_central_bin_size_in_cm;
  bool is_arccorrected;
};

END_NAMESPACE_STIR

#endif // __stir_INTERFILEHEADER_H__
