// 
// $Id$: $Date$
//
#ifndef __Interfile_h__
#define __Interfile_h__

#include "KeyParser.h"
#include "NumericInfo.h"
#include "pet_common.h"
#include "imagedata.h"

// This reads the first image from an Interfile header
// If there is trouble interpreting the header, 
// ask_image_details() is called instead
// KT 14/10/98 make arg istream
PETImageOfVolume read_interfile_image(istream& input);

// KT 13/11/98 new
// This first opens a stream and then calls the previous function
PETImageOfVolume read_interfile_image(char *filename);

// This outputs an Interfile header and data for a Tensor3D object.
// Extensions .hv and .v will be added to the parameter 'filename' 
// Return is 'true' when succesful, 'false' otherwise.
// 
// In fact, at the moment 2 headers are output:
// .hv is 'new-style' (suitable for Mediman for instance), 
// .ahv is 'old-style' (suitable for Analyze for instance)
// They both point to the same file with binary data.
// KT 09/10/98 added const for image arg
// KT 09/11/98 added voxel_size & output_type
template <class NUMBER>
bool write_basic_interfile(const char * const filename, 
			   const Tensor3D<NUMBER>& image,
			   const Point3D& voxel_size,
			   const NumericType output_type = NumericType::FLOAT);

// KT 09/11/98 added this for 'old' interface
template <class NUMBER>
inline bool 
write_basic_interfile(const char * const filename, 
		      const Tensor3D<NUMBER>& image,
		      const NumericType output_type = NumericType::FLOAT)
{
  return
    write_basic_interfile(filename, 
			  image, 
			  Point3D(1,1,1), 
			  output_type);
}

// KT 09/11/98 new to preserve voxel_size
bool 
write_basic_interfile(const char * const filename, 
		      const PETImageOfVolume& image,
		      const NumericType output_type = NumericType::FLOAT);


// This reads the first 3D sinogram from an Interfile header
// If there is trouble interpreting the header, 
// ask_PSOV_details() is called instead
// KT 26/10/98 new
PETSinogramOfVolume read_interfile_PSOV(istream& input);

// KT 13/11/98 new
// This first opens a stream and then calls the previous function
PETSinogramOfVolume read_interfile_PSOV(char *filename);

#endif // __Interfile_h__

