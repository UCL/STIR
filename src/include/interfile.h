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
PETImageOfVolume read_interfile_image(fstream& input);

// This outputs an Interfile header and data for a Tensor3D object.
// Extensions .hv and .v will be added to the parameter 'filename' 
// Return is 'true' when succesful, 'false' otherwise.


template <class NUMBER>
bool write_basic_interfile(const char * const filename, Tensor3D<NUMBER>& image);

#endif // __Interfile_h__
