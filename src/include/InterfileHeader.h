//
// $Id$ :$Date$
//

#ifndef __INTERFILEHEADER_H__
#define __INTERFILEHEADER_H__

// #include <fstream> // KT 03/11/98 not used anymore
// KT 20/06/98 used for type_of_numbers and byte_order
#include "NumericInfo.h"
#include "KeyParser.h"

class InterfileHeader : public KeyParser
{
public:
  //KT 26/10/98 moved to .cxx file
  InterfileHeader(istream& f);

protected:

  //KT 26/10/98 removed virtual void init_keys();
  // If you override the next function, call this one first
  // Returns 0 of OK, 1 of not.
  virtual int post_processing();


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

  ASCIIlist_type PET_data_type_values;	
  int PET_data_type_index;

  // TODO these shouldn't be here, but in PETStudy or something

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
   InterfileImageHeader(istream& f)
     : InterfileHeader(f)
   {}

protected:

  // Returns 0 of OK, 1 of not.
  virtual int post_processing();

};

#if 1
#include "sinodata.h"

class InterfilePSOVHeader : public InterfileHeader
{
public:
   //KT 26/10/98 
  InterfilePSOVHeader(istream& f);

protected:

  //KT 26/10/98 virtual void init_keys();
  // Returns 0 of OK, 1 of not.
  virtual int post_processing();

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

private:
  void resize_segments_and_set();
  int find_storage_order();

};
	
#endif

#endif // __INTERFILEHEADER_H__
