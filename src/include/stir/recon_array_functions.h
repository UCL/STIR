//
// $Id$
//
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, Hammersmith Imanet Ltd
    This file is part of STIR.

    This file is free software; you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation; either version 2.1 of the License, or
    (at your option) any later version.

    This file is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    See STIR/LICENSE.txt for details
*/

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

#include "stir/common.h"

START_NAMESPACE_STIR


// TODO template this all in data type of Viewgram et al


template <typename elemT> class SegmentByView;
template <typename elemT> class SegmentBySinogram;
template <typename elemT> class Viewgram;
template <typename elemT> class RelatedViewgrams;
template <int num_dimensions, typename elemT> class DiscretisedDensity;

#if 0
//! scales an image and adds it to another
void multiply_and_add(DiscretisedDensity<3,float> &image_res, const DiscretisedDensity<3,float> &image_scaled, float scalar);
#endif

//! truncates negative values to zero
float neg_trunc(float x);

//MJ 17/05/2000 disabled
// gives 1 if negative, 0 otherwise
//float neg_indicate(float x){return (x<=0.0)?1.0:0.0;}

#if 0
//! sets non-positive voxel values to small positive ones AND truncate to circular FOV
/*! \warning Really makes only sense for images of type DiscretisedDensityOnCartesianGrid
  (or derived types)
 */
void threshold_min_to_small_positive_value_and_truncate_rim(DiscretisedDensity<3,float>& input_image, const int rim_truncation_image=0);

//! divide sinograms and set 'edge bins' to zero, put answer in numerator
void divide_and_truncate(DiscretisedDensity<3,float>& numerator, 
			 const DiscretisedDensity<3,float>& denominator,
			 const int rim_truncation,
			 int & count);
#endif

//! divide viewgrams and set 'edge bins' to zero, put answer in numerator
void divide_and_truncate(Viewgram<float>& numerator, const Viewgram<float>& denominator,
			 const int rim_truncation_sino,
			 int& count, int& count2, double* f = NULL);

//! divide related viewgrams and set 'edge bins' to zero, put answer in numerator
void divide_and_truncate(RelatedViewgrams<float>& numerator, const RelatedViewgrams<float>& denominator,
			 const int rim_truncation_sino,
			 int& count, int& count2, double* f = NULL);

//! sets to zero voxels within rim_truncation_image of the FOV rim
void truncate_rim(DiscretisedDensity<3,float>& image_input, 
		  const int rim_truncation_image,
		  const bool strictly_less_than_radius = true);



//! sets the first and last rim_truncation_sino bins at the 'edges' to zero
void truncate_rim(SegmentByView<float>& seg, const int rim_truncation_sino);

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
void accumulate_loglikelihood(Viewgram<float>& projection_data, 
			 const Viewgram<float>& estimated_projections,
			 const int rim_truncation_sino,
			 double* accum);


END_NAMESPACE_STIR
#endif // __recon_array_functions_h_

