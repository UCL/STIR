//
// $Id$ :$Date$
//

#ifndef __INTERFILEHEADER_H__
#define __INTERFILEHEADER_H__

#include <fstream>
// KT 20/06/98 used for type_of_numbers and byte_order
#include "NumericInfo.h"
#include "KeyParser.h"

class InterfileHeader : public KeyParser
{
public:
   InterfileHeader(istream& f)
     : KeyParser(f)
   {}

protected:

  // If you override the next two functions, call these first
  virtual void init_keys();
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
  int	bytes_per_pixel;
  String data_file_name;


  void ReadMatrixInfo();
  void ReadFramesInfo();
  void OpenFileStream();

public :

  ASCIIlist_type PET_data_type_values;	
  int PET_data_type_index;

  // TODO these shouldn't be here, but in PETStudy or something

  // 'Final' variables
  fstream*		in_stream;
  int			max_r_index;
  int			storage_order;
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
  vector<String>	matrix_labels;
  vector<IntVect>	matrix_size;
  DoubleVect		pixel_sizes;
  IntVect		sqc;
  // KT 01/08/98 changed to DoubleVect
  DoubleVect		image_scaling_factor;
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

#if 0
class InterfilePSOVHeader : public InterfileHeader
{
public:
   InterfilePSOVHeader(istream& f)
     : InterfileHeader(f)
   {}

protected:

  // Returns 0 of OK, 1 of not.
  virtual int post_processing();

public:
  PETSinogramOfVolume::StorageOrder storage_order;
};
	
#endif

#endif // __INTERFILEHEADER_H__
