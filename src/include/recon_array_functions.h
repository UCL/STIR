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

// AZ 04/10/99: added
void divide_and_truncate(const int view, // const int view45,
			 PETSegmentBySinogram& numerator, const PETSegmentBySinogram& denominator,
			 const int rim_truncation_sino,
			 int& count, int& count2, float* f = NULL);

// AZ 04/10/99: added
void divide_and_truncate(PETViewgram& numerator, const PETViewgram& denominator,
			 const int rim_truncation_sino,
			 int& count, int& count2, float* f = NULL);

// AZ 07/10/99: added

void truncate_rim(PETSegmentBySinogram& seg, const int rim_truncation_sino, const int view);

// AZ 07/10/99: added

void truncate_rim(PETViewgram& viewgram, const int rim_truncation_sino);

void truncate_end_planes(PETImageOfVolume &input_image);

void divide_array(PETSegmentByView& numerator,PETSegmentByView& denominator);


//MJ 03/01/2000  Trying to adhoc parallelize a loglikelihood computation

void accumulate_loglikelihood(const int view,
			 PETSegmentBySinogram& projection_data, 
			 const PETSegmentBySinogram& estimated_projections,
			 const int rim_truncation_sino, float *accum);

void accumulate_loglikelihood(PETViewgram& projection_data, 
			 const PETViewgram& estimated_projections,
			 const int rim_truncation_sino,
			 float* accum);


#endif // __recon_array_functions_h_

