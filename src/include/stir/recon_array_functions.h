//
// $Id$
//

#ifndef __stir_recon_array_functions_h_
#define __stir_recon_array_functions_h_

/*!
  \file 
  \ingroup buildblock
 
  \brief a variety of useful functions

  \author Matthew Jacobson
  \author Kris Thielemans
  \author PARAPET project

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/


#include "stir/common.h"

START_NAMESPACE_STIR


// TODO template this all in data type of Viewgram et al


template <typename elemT> class SegmentByView;
template <typename elemT> class SegmentBySinogram;
template <typename elemT> class Viewgram;
template <typename elemT> class RelatedViewgrams;
template <int num_dimensions, typename elemT> class DiscretisedDensity;


//! scales an image and adds it to another
void multiply_and_add(DiscretisedDensity<3,float> &image_res, const DiscretisedDensity<3,float> &image_scaled, float scalar);

//! truncates negative values to zero
float neg_trunc(float x);

//MJ 17/05/2000 disabled
// gives 1 if negative, 0 otherwise
//float neg_indicate(float x){return (x<=0.0)?1.0:0.0;}

//! sets non-positive voxel values to small positive ones
void threshold_min_to_small_positive_value(DiscretisedDensity<3,float>& input_image, const int rim_truncation_image=0);

//! divide sinograms and set 'edge bins' to zero, put answer in numerator
void divide_and_truncate(DiscretisedDensity<3,float>& numerator, 
			 const DiscretisedDensity<3,float>& denominator,
			 const int rim_truncation,
			 int & count);

#if 0
// AZ 04/10/99: added
//! divide sinograms and set 'edge bins' to zero, put answer in numerator
void divide_and_truncate(const int view, // const int view45,
			 SegmentBySinogram<float>& numerator, const SegmentBySinogram<float>& denominator,
			 const int rim_truncation_sino,
			 int& count, int& count2, float* f = NULL);

#endif

//! divide viewgrams and set 'edge bins' to zero, put answer in numerator
void divide_and_truncate(Viewgram<float>& numerator, const Viewgram<float>& denominator,
			 const int rim_truncation_sino,
			 int& count, int& count2, float* f = NULL);

//! divide related viewgrams and set 'edge bins' to zero, put answer in numerator
void divide_and_truncate(RelatedViewgrams<float>& numerator, const RelatedViewgrams<float>& denominator,
			 const int rim_truncation_sino,
			 int& count, int& count2, float* f = NULL);


//! sets to zero voxels within rim_truncation_image of the FOV rim
void truncate_rim(DiscretisedDensity<3,float>& image_input, 
		  const int rim_truncation_image);



//! sets the first and last rim_truncation_sino bins at the 'edges' to zero
void truncate_rim(SegmentByView<float>& seg, const int rim_truncation_sino);
//! sets the first and last rim_truncation_sino bins at the 'edges' to zero
void truncate_rim(SegmentBySinogram<float>& seg, const int rim_truncation_sino);
#if 0
//! sets the first and last rim_truncation_sino bins at the 'edges' to zero
void truncate_rim(SegmentBySinogram<float>& seg, const int rim_truncation_sino, const int view);
#endif

//! sets the first and last rim_truncation_sino bins at the 'edges' to zero
void truncate_rim(Viewgram<float>& viewgram, const int rim_truncation_sino);

//! sets the end planes of the image to zero
void truncate_end_planes(DiscretisedDensity<3,float> &input_image, int input_num_planes=1);

//! simple division of two sinograms, x/0 = 0
void divide_array(SegmentByView<float>& numerator, const SegmentByView<float>& denominator);

//! simple division of two images, x/0 = 0
void divide_array(DiscretisedDensity<3,float>& numerator, const DiscretisedDensity<3,float>& denominator);

//MJ 03/01/2000  Trying to adhoc parallelize a loglikelihood computation

//! compute the log term of the loglikelihood function for given part of the projection space
void accumulate_loglikelihood(const int view,
			 SegmentBySinogram<float>& projection_data, 
			 const SegmentBySinogram<float>& estimated_projections,
			 const int rim_truncation_sino, float *accum);
//! compute the log term of the loglikelihood function for given part of the projection space
void accumulate_loglikelihood(Viewgram<float>& projection_data, 
			 const Viewgram<float>& estimated_projections,
			 const int rim_truncation_sino,
			 float* accum);

float min_positive_value(DiscretisedDensity<3,float>& input_image, const int rim_truncation_image);

END_NAMESPACE_STIR
#endif // __recon_array_functions_h_

