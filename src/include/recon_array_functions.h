//
// $Id$: $Date$
//

#ifndef __recon_array_functions_h_
#define __recon_array_functions_h_


#include "sinodata.h"
//#include "PETScannerInfo.h"
#include "imagedata.h"

// KT 09/11/98 new
void truncate_rim(PETSegmentByView& seg, const int rim_truncation_sino);
void truncate_rim(PETSegmentBySinogram& seg, const int rim_truncation_sino);
void truncate_rim(PETImageOfVolume& image_input, 
		  const int rim_truncation_image);

void multiply_and_add(PETImageOfVolume &image_res, const PETImageOfVolume &image_scaled, float scalar);

float neg_trunc(float x);

void divide_and_truncate(PETImageOfVolume& numerator, 
			 const PETImageOfVolume& denominator,
			 const int rim_truncation,
			 int & count);

void divide_and_truncate_den(const PETImageOfVolume& numerator, 
			 PETImageOfVolume& denominator,
			 const int rim_truncation,
			 int & count);

void truncate_end_planes(PETImageOfVolume &input_image);

#endif // __recon_array_functions_h_

