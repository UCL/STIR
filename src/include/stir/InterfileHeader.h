//
// $Id$
//
/*! 
  \file
  \ingroup buildblock 
  \brief  This file declares the classes InterfileHeader,
          InterfileImageHeader, InterfilePDFSHeader  

  \author Kris Thielemans
  \author Sanida Mustafovic
  \author PARAPET project

  $Date$
  $Revision$

  \warning Different datasets in 1 header are not yet supported.

  See http://www.irsl.org/~kris for a description of the full
  proposal for Interfile headers for 3D PET.
*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/

#ifndef __INTERFILEHEADER_H__
#define __INTERFILEHEADER_H__

#include "stir/ByteOrder.h"
#include "stir/NumericInfo.h"
#include "stir/KeyParser.h"

#include "stir/ProjDataFromStream.h"



START_NAMESPACE_STIR

class ProjDataInfo;

/*!
  \brief a class for Interfile keywords (and parsing) common to 
  all types of data
  \ingroup buildblock
  */
class InterfileHeader : public KeyParser
{
public:
  InterfileHeader();

  virtual ~InterfileHeader() {}

protected:

  // Returns false if OK, true if not.
  virtual bool post_processing();


private:

  // Lists of possible values for some keywords
  ASCIIlist_type number_format_values;	
  ASCIIlist_type byte_order_values;
  ASCIIlist_type type_of_data_values;

  // Corresponding variables here

  int number_format_index;
  int byte_order_index;
  int type_of_data_index;

  // Extra private variables which will be translated to something more useful
  int bytes_per_pixel;

  // Louvain la Neuve style of 'image scaling factors'
  double lln_quantification_units;

  void read_matrix_info();
  void read_frames_info();

public :

  string originating_system;
  
  ASCIIlist_type PET_data_type_values;	
  int PET_data_type_index;

  // TODO these shouldn't be here, but in PETStudy or something

  // 'Final' variables

  string data_file_name;

  //! This will be determined from number_format_index and bytes_per_pixel
  NumericType		type_of_numbers;
  //! This will be determined from byte_order_index, or just keep its default value;
  ByteOrder file_byte_order;
	
  int			num_dimensions;
  int			num_time_frames;
  vector<string>	matrix_labels;
  vector<IntVect>	matrix_size; 
  DoubleVect		pixel_sizes;
  // KT 03/11/98 cannot remove 'sqc' because of VC++ compiler bug (it complains about matrix_size.resize(1))
  IntVect		sqc; 
  vector<DoubleVect>	image_scaling_factors;
  UlongVect		data_offset;
};


/*!
  \brief a class for Interfile keywords (and parsing) specific to images
  \ingroup buildblock
  */
class InterfileImageHeader : public InterfileHeader
{
public:
  InterfileImageHeader()
     : InterfileHeader()
   {}

protected:

  //! Returns false if OK, true if not.
  virtual bool post_processing();

};

/*!
  \brief a class for Interfile keywords (and parsing) specific to 
  projection data (i.e. ProjDataFromStream)
  \ingroup buildblock
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
  int num_rings;
  int num_detectors_per_ring;
  
  double transaxial_FOV_diameter_in_cm;
  double ring_diameter_in_cm;
  double distance_between_rings_in_cm;
  double bin_size_in_cm;
  // this is in degrees
  double view_offset;


  bool is_arccorrected;
};

END_NAMESPACE_STIR

#endif // __INTERFILEHEADER_H__
