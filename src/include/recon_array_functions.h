//
// $Id$
//

#ifndef __recon_array_functions_h_
#define __recon_array_functions_h_

/*!
  \file 
  \ingroup buildblock
 
  \brief a variety of useful functions

  \author Matthew Jacobson
  \author PARAPET project

  \date    $Date$
  \version $Revision$
*/


#include "sinodata.h"
#include "imagedata.h"

START_NAMESPACE_TOMO


// KT 09/11/98 new

//! sets the first and last rim_truncation_sino bins at the 'edges' to zero
void truncate_rim(PETSegmentByView& seg, const int rim_truncation_sino);
//! sets the first and last rim_truncation_sino bins at the 'edges' to zero
void truncate_rim(PETSegmentBySinogram& seg, const int rim_truncation_sino);

//! sets to zero voxels within rim_truncation_image of the FOV rim
void truncate_rim(PETImageOfVolume& image_input, 
		  const int rim_truncation_image);

//! scales an image and adds it to another
void multiply_and_add(PETImageOfVolume &image_res, const PETImageOfVolume &image_scaled, float scalar);

//! truncates negative values to zero
float neg_trunc(float x);

//MJ 17/05/2000 disabled
// gives 1 if negative, 0 otherwise
//float neg_indicate(float x){return (x<=0.0)?1.0:0.0;}

//! sets non-positive voxel values to small positive ones
void set_negatives_small(PETImageOfVolume& input_image, const int rim_truncation_image=0);

//! divide sinograms and set 'edge bins' to zero, put answer in numerator
void divide_and_truncate(PETImageOfVolume& numerator, 
			 const PETImageOfVolume& denominator,
			 const int rim_truncation,
			 int & count);

// AZ 04/10/99: added
//! divide sinograms and set 'edge bins' to zero, put answer in numerator
void divide_and_truncate(const int view, // const int view45,
			 PETSegmentBySinogram& numerator, const PETSegmentBySinogram& denominator,
			 const int rim_truncation_sino,
			 int& count, int& count2, float* f = NULL);

// AZ 04/10/99: added

//! divide sinograms and set 'edge bins' to zero, put answer in numerator
void divide_and_truncate(PETViewgram& numerator, const PETViewgram& denominator,
			 const int rim_truncation_sino,
			 int& count, int& count2, float* f = NULL);

// AZ 07/10/99: added
//! sets the first and last rim_truncation_sino bins at the 'edges' to zero
void truncate_rim(PETSegmentBySinogram& seg, const int rim_truncation_sino, const int view);

// AZ 07/10/99: added
//! sets the first and last rim_truncation_sino bins at the 'edges' to zero
void truncate_rim(PETViewgram& viewgram, const int rim_truncation_sino);

//! sets the end planes of the image to zero
void truncate_end_planes(PETImageOfVolume &input_image, int plane_truncation=1);

//! simple division of two sinograms, x/0 = 0
void divide_array(PETSegmentByView& numerator,PETSegmentByView& denominator);

//! simple division of two images, x/0 = 0
void divide_array(PETImageOfVolume& numerator,PETImageOfVolume& denominator);

//MJ 03/01/2000  Trying to adhoc parallelize a loglikelihood computation

//! compute the log term of the loglikelihood function for given part of the projection space
void accumulate_loglikelihood(const int view,
			 PETSegmentBySinogram& projection_data, 
			 const PETSegmentBySinogram& estimated_projections,
			 const int rim_truncation_sino, float *accum);
//! compute the log term of the loglikelihood function for given part of the projection space
void accumulate_loglikelihood(PETViewgram& projection_data, 
			 const PETViewgram& estimated_projections,
			 const int rim_truncation_sino,
			 float* accum);

END_NAMESPACE_TOMO
#endif // __recon_array_functions_h_

