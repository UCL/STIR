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
PETImageOfVolume read_interfile_image(fstream& input);

// This outputs an Interfile header and data for a Tensor3D object.
// Extensions .hv and .v will be added to the parameter 'filename' 
// Return is 'true' when succesful, 'false' otherwise.
// 
// In fact, at the moment 2 headers are output:
// .hv is 'new-style' (suitable for Mediman for instance), 
// .ahv is 'old-style' (suitable for Analyze for instance)
// They both point to the same file with binary data.
// KT 09/10/98 added const for image arg
template <class NUMBER>
bool write_basic_interfile(const char * const filename, const Tensor3D<NUMBER>& image);

#endif // __Interfile_h__
