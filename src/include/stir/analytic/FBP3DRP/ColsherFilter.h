//
//

/*! 
  \file 
  \ingroup FBP3DRP
  \brief Colsher filter class
  \author Claire LABBE
  \author Kris Thielemans
  \author PARAPET project
*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2004, Hammersmith Imanet Ltd

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

#ifndef __ColsherFilter_H__
#define __ColsherFilter_H__

#ifdef NRFFT
#include "stir_experimental/Filter.h"
#else
#include "stir/ArrayFilterUsingRealDFTWithPadding.h"
#include "stir/TimedObject.h"
#endif

START_NAMESPACE_STIR
#ifdef NRFFT
template <typename elemT> class Viewgram;
#endif

/*!
  \ingroup FBP3DRP
  \brief This class contains the Colsher filter used for 3D-PET reconstruction.

  The Colsher filter is combined with a 2-dimensional apodising Hamming filter.
*/
class ColsherFilter: 
#ifdef NRFFT
  public Filter2D <float>
#else
  public ArrayFilterUsingRealDFTWithPadding<2,float>,
  public TimedObject
#endif
{

public:
#ifndef NRFFT       
  //! Default constructor
  /*! \warning Leaves object in ill-defined state*/
  ColsherFilter() {}
  /*!
    \brief constructor for the ColsherFilter.
    
    \param theta_max
           the polar angle corresponding to the maximum oblique angle 
	   included in the reconstruction.

    The \c alpha and \c fc parameters are designed to minimize the 
    amplification of noise.

    The \c stretch_factor parameters can be used to define the Colsher
    filter via a finer grid to avoid the problems with sampling a 
    continuous filter in frequency space. For the ramp-filter, this can 
    be done using analytic integration, but here we have to do it numerically. 
  */
  explicit ColsherFilter(float theta_max,
			 float alpha_colsher_axial=1.F, float fc_colsher_axial=0.5F,
			 float alpha_colsher_radial=1.F, float fc_colsher_radial=0.5F,
			 const int stretch_factor_axial=2,
			 const int stretch_factor_planar=2);
  //! Initialise filter values
  /*! creates a 2D Colsher filter of size height*width,
    \param theta the polar angle
    \param d_a the sampling distance in the 's' coordinate
    \param d_b the sampling distance in the 't' coordinate
  */
  Succeeded
    set_up(int height, int width, float theta,
	   float d_a, float d_b);
#else
  ColsherFilter(int height, int width, float gamma, float theta_max,
		  float d_a, float d_b,
                  float alpha_colsher_axial, float fc_colsher_axial,
                  float alpha_colsher_radial, float fc_colsher_radial);
#endif

  virtual std::string parameter_info() const;
  
  ~ColsherFilter() {}


  private:
#ifdef NRFFT
    //! gamma is the polar angle
    float gamma;
        /* d_a, d_b are used to convert to millimeter.*/
    //! d_a is initialised with the sampling distance
    float d_a;
    //! d_b is initialised with ring spacing * sin(theta)
    float d_b; 
#endif
    //! theta_max corresponds to the maximum aperture in the reconstruction 
    float theta_max;
    //! value of the axial alpha parameter for apodized Hamming window
    float alpha_axial;
    //! value of the axial cut-off frequency of the Colsher filter
    float fc_axial;
    //! value of the planar alpha parameter for apodized Hamming window
    float alpha_planar;
    //! value of the planar cut-off frequency of the Colsher filter
    float fc_planar;
    int stretch_factor_axial;
    int stretch_factor_planar;
};

#ifdef NRFFT
void Filter_proj_Colsher(Viewgram<float> & view_i,
			 Viewgram<float> & view_i1,
                         ColsherFilter& CFilter, 
                         int PadS, int PadZ);
#endif
END_NAMESPACE_STIR

#endif //  __ColsherFilter_H__

