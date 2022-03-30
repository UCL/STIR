/*
    Copyright (C) 2002-2007, Hammersmith Imanet Ltd
    Copyright (C) 2013, 2016, 2018, 2020 University College London
    Copyright 2017 ETH Zurich, Institute of Particle Physics and Astrophysics
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0 AND License-ref-PARAPET-license

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
  \author Parisa Khateri

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
#include "stir/date_time_functions.h"

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
    //! Get a shared pointer to the exam information
    shared_ptr<const ExamInfo>
      get_exam_info_sptr() const;
    //! Get the exam information
    const ExamInfo&
      get_exam_info() const;

    std::string version_of_keys;

    std::string siemens_mi_version;
  protected:
    //! will be called when the version keyword is found
    /*! This callback function provides an opportunity to change the keymap depending on the version
        (which can be obtained from \c version_of_keys).

        Just calls \c set_variable().

        It is expected that if this is function is re-implemented in a derived class, it calls the
        base-class version.
    */
    virtual void set_version_specific_keys();
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
  
  std::string isotope_name;
  float calibration_factor;
private:

  // Louvain la Neuve style of 'image scaling factors'
  double lln_quantification_units;

  
 protected:
  //! Overload with specifics for STIR3.0 for backwards compatibility
  virtual void set_version_specific_keys();
  virtual void read_matrix_info();
  virtual void read_num_energy_windows();
  void read_frames_info();
  //! \brief Get the number of datasets
  /*! To be overloaded by derived classes if multiple "dimensions" are supported.
      Default is just to use num_time_frames.
  */
  virtual int get_num_datasets() const { return num_time_frames; }

public :

  ASCIIlist_type type_of_data_values;
  int type_of_data_index;

  ASCIIlist_type PET_data_type_values;	
  int PET_data_type_index;

  ASCIIlist_type process_status_values;
  int process_status_index;

  // 'Final' variables

  std::string data_file_name;

  DateTimeStrings study_date_time;

  //! This will be determined from number_format_index and bytes_per_pixel
  NumericType		type_of_numbers;
  //! This will be determined from byte_order_index, or just keep its default value;
  ByteOrder file_byte_order;
	
  int			num_dimensions;
  int			num_energy_windows;
  std::vector<std::string>	matrix_labels;
  std::vector<std::vector<int> > matrix_size; 
  std::vector<float>	pixel_sizes;
  std::vector<std::vector<double> > image_scaling_factors;
  std::vector<unsigned long> data_offset_each_dataset;

  // Acquisition parameters
  //!
  //! \brief lower_en_window_thresholds
  //! \details Low energy window limit
  std::vector<float> lower_en_window_thresholds;

  //!
  //! \brief upper_en_window_thresholds
  //! \details High energy window limit
  std::vector<float> upper_en_window_thresholds;
  // end acquisition parameters
  
 protected:
  // version 3.3 had only a single offset. we'll internally replace it with data_offset_each_dataset
  unsigned long data_offset;

  float bed_position_horizontal;
  float bed_position_vertical;
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
  //! \brief Get the number of datasets
  //! \details no. time frames * no. data types (kinetic params) * no. gates
  //! Currently, this is only implemented for either multiple time frames OR multiple data types (gates not considered).
  virtual int get_num_datasets() const { return num_time_frames*num_image_data_types; }

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
 
  // derived values
  int num_segments;
  int num_views;
  int num_bins;
  ProjDataFromStream::StorageOrder storage_order;
  shared_ptr<ProjDataInfo> data_info_sptr;

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
  
  //! \name new variables for block geometry
  //@{
  std::string scanner_geometry;
  float axial_distance_between_crystals_in_cm;
  float transaxial_distance_between_crystals_in_cm;
  float axial_distance_between_blocks_in_cm;
  float transaxial_distance_between_blocks_in_cm;
  //@}
  
  //! \name new variables for generic geometry
  //@{
  std::string crystal_map;
  //@}
  // end scanner parameters

  double effective_central_bin_size_in_cm;
  bool is_arccorrected;
};

END_NAMESPACE_STIR

#endif // __stir_INTERFILEHEADER_H__
