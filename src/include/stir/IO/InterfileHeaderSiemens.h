/*
    Copyright (C) 2002-2007, Hammersmith Imanet Ltd
    Copyright (C) 2013, 2016 University College London
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
  \brief  This file declares the classes stir::InterfileHeaderSiemens,
          stir::InterfileImageHeader, stir::InterfilePDFSHeader  

  \author Kris Thielemans
  \author Sanida Mustafovic
  \author PARAPET project

  See http://stir.sourceforge.net for a description of the full
  proposal for Interfile headers for 3D PET.
*/


#ifndef __stir_InterfileHeaderSiemens_H__
#define __stir_InterfileHeaderSiemens_H__

#include "stir/ByteOrder.h"
#include "stir/NumericInfo.h"
#include "stir/IO/InterfileHeader.h"
#include "stir/ProjDataFromStream.h"
#include "stir/ExamInfo.h"

START_NAMESPACE_STIR

class ProjDataInfo;

/*!
  \brief a class for Interfile keywords (and parsing) common to 
  all types of data
  \ingroup InterfileIO
  */
class InterfileHeaderSiemens : public InterfileHeader
{
public:
  //! A value that can be used to signify that a variable has not been set during parsing.
  static const double double_value_not_set;

  InterfileHeaderSiemens();

  virtual ~InterfileHeaderSiemens() {}

protected:
  // Returns false if OK, true if not.
  virtual bool post_processing();

private:

  // TODO the next few ones should be made static members
  // Lists of possible values for some keywords
  //ASCIIlist_type number_format_values;	
  ASCIIlist_type byte_order_values;
  ASCIIlist_type patient_position_values;
  
  // Corresponding variables here

  //int number_format_index;
  int byte_order_index;
  int patient_position_index;

  void set_type_of_data();

 protected:
  void read_scan_data_types();


};

#if 0 // probably not necessary
/*!
  \brief a class for Interfile keywords (and parsing) specific to images
  \ingroup InterfileIO
  */
class InterfileImageHeader : public InterfileHeaderSiemens
{
 private:
  typedef InterfileHeaderSiemens base_type;

public:
  InterfileImageHeader();
  std::vector<double>	first_pixel_offsets;

protected:
  virtual void read_matrix_info();
  //! Returns false if OK, true if not.
  virtual bool post_processing();

};

#endif

/*!
\brief a class for Interfile keywords (and parsing) specific to
Siemens PET projection or list mode data
\ingroup InterfileIO
*/
class InterfileRawDataHeaderSiemens : public InterfileHeaderSiemens
  {
  public:
    InterfileRawDataHeaderSiemens();

  protected:

    //! Returns false if OK, true if not.
    virtual bool post_processing();
    // need this to be false for the listmode data
    bool is_arccorrected;
  public:

    ProjDataFromStream::StorageOrder storage_order;
    std::vector<int> segment_sequence;
    shared_ptr<ProjDataInfo> data_info_ptr;

  private:
    void resize_segments_and_set();
    //void read_frames_info();
    void read_num_energy_windows();

    //int find_storage_order();

    std::vector<float> lower_en_window_thresholds;
    std::vector<float> upper_en_window_thresholds;

  protected:

    int axial_compression;
    int maximum_ring_difference;
    int num_energy_windows;

    std::vector<int> segment_table;
    int num_segments;
    int num_rings;
    int num_views;
    int num_bins;
    int num_tof_bins;
  };


/*!
\brief a class for Interfile keywords (and parsing) specific to
projection data (i.e. ProjDataFromStream)
\ingroup InterfileIO
*/
class InterfilePDFSHeaderSiemens : public InterfileRawDataHeaderSiemens
  {
  public:
    InterfilePDFSHeaderSiemens();

  protected:

    //! Returns false if OK, true if not.
    virtual bool post_processing();

  public:

    std::vector<std::string> applied_corrections;
    bool compression;

  private:
    void resize_segments_and_set();

    int find_storage_order();

    int num_scan_data_types;
    std::vector<std::string> scan_data_types;
    void read_scan_data_types();
    int total_num_sinograms;
    std::string compression_as_string;

    int num_buckets;
    std::vector<int> bucket_singles_rates;
    void read_bucket_singles_rates();
  };

/*!
\brief a class for Interfile keywords (and parsing) specific to
Siemesn listmode data (in PETLINK format)
\ingroup InterfileIO
*/
class InterfileListmodeHeaderSiemens : public InterfileRawDataHeaderSiemens
  {
  public:
    InterfileListmodeHeaderSiemens();

  protected:

    //! Returns false if OK, true if not.
    virtual bool post_processing();

  public:
    //! Get axial compression
    int get_axial_compression() const ;
    //! Get the maximum ring difference
    int get_maximum_ring_difference() const;
    //! Get the num of views
    int get_num_views() const;
    //! Gat the num of projections
    int get_num_projections() const;

 
  private:

    int find_storage_order();

  };

END_NAMESPACE_STIR

#endif // __stir_InterfileHeaderSiemens_H__
