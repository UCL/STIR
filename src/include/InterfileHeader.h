//
// $Id$ 
//

#ifndef __INTERFILEHEADER_H__
#define __INTERFILEHEADER_H__

// KT 15/12/99 added
#include "ByteOrder.h"
#include "NumericInfo.h"
#include "KeyParser.h"

class InterfileHeader : public KeyParser
{
public:
  // KT 13/11/98 moved stream arg from constructor to parse()
  InterfileHeader();

protected:

  //KT 26/10/98 removed virtual void init_keys();
  // If you override the next function, call this one first
  // Returns 0 of OK, 1 of not.
  virtual bool post_processing();


private:

  // Lists of possible values for some keywords
  // KT 20/06/98 new
  ASCIIlist_type number_format_values;	
  // KT 01/08/98 new
  ASCIIlist_type byte_order_values;
  ASCIIlist_type type_of_data_values;

  // Corresponding variables here

  int number_format_index;
  int byte_order_index;
  int type_of_data_index;

  // Extra private variables which will be translated to something more useful
  int bytes_per_pixel;

  void read_matrix_info();
  void read_frames_info();

public :

  // KT 26/11/98 new
  string originating_system;
  
  ASCIIlist_type PET_data_type_values;	
  int PET_data_type_index;

  // TODO these should not be here, but in PETStudy or something

  // 'Final' variables

  //KT 26/10/98 moved here
  string data_file_name;

  //KT 26/10/98 removed   fstream*		in_stream;
  // KT 20/06/98 new
  // This will be determined from number_format_index and bytes_per_pixel
  NumericType		type_of_numbers;
  // KT 01/08/98 new
  // This will be determined from byte_order_index, or just keep
  // its default value;
  ByteOrder file_byte_order;
	
  // KT 01/08/98 changed name to num_dimensions and num_time_frames
  int			num_dimensions;
  int			num_time_frames;
  vector<string>	matrix_labels;
  vector<IntVect>	matrix_size;
  DoubleVect		pixel_sizes;
  // KT 03/11/98 cannot remove 'sqc' because of VC++ compiler bug (it complains about matrix_size.resize(1))
  IntVect		sqc; 
  // KT 29/10/98 changed to vector<DoubleVect>
  vector<DoubleVect>	image_scaling_factors;
  // KT 01/08/98 changed to UlongVect
  UlongVect		data_offset;
};


class InterfileImageHeader : public InterfileHeader
{
public:
  // KT 13/11/98 moved stream arg from constructor to parse()
  InterfileImageHeader()
     : InterfileHeader()
   {}

protected:

  // Returns 0 of OK, 1 of not.
  virtual bool post_processing();

};

#include "sinodata.h"

class InterfilePSOVHeader : public InterfileHeader
{
public:
   //KT 26/10/98 
  // KT 13/11/98 moved stream arg from constructor to parse()
  InterfilePSOVHeader();

protected:

  //KT 26/10/98 virtual void init_keys();
  // Returns 0 of OK, 1 of not.
  virtual bool post_processing();

public:
  vector<int> segment_sequence;
  vector<int> min_ring_difference; 
  vector<int> max_ring_difference; 
  vector<int> num_rings_per_segment;
  
  // derived values
  int num_segments;
  int num_views;
  int num_bins;
  PETSinogramOfVolume::StorageOrder storage_order;
  // KT 26/11/98 new
  PETScanInfo scan_info;

private:
  void resize_segments_and_set();
  int find_storage_order();

  // KT 26/11/98 new

  // members that will be used to set scan_info
  int num_rings;
  int num_detectors_per_ring;
  // these 4 distances are in cm
  double transaxial_FOV_diameter_in_cm;
  // KT 31/03/99 new
  double ring_diameter_in_cm;
  double distance_between_rings_in_cm;
  double bin_size_in_cm;
  // this is in degrees
  double view_offset;

};

#endif // __INTERFILEHEADER_H__
