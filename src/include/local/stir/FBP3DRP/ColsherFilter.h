//
// $Id$
//

/*! 
  \file 
  \brief Colsher filter class
  \author Claire LABBE
  \author PARAPET project
  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/

#ifndef __ColsherFilter_H__
#define __ColsherFilter_H__

#include "local/stir/Filter.h"

START_NAMESPACE_STIR

template <typename elemT> class Segment;
template <typename elemT> class Viewgram;


/*!

  \brief This class contains the Colsher filter and is derived from Filter2D class. 
  The Colsher filter is combined with a 2-dimensional apodising Hamming filter.
*/
class ColsherFilter: public Filter2D <float>
{

public:
       
        /*!
          \brief constructor for the ColsherFilter.

          This default constructor creates a 2D Colsher filter of size height*width,
          and takes for defining the Colsher filter parameters: theta_max,
          the co-polar angle corresponding to the maximum oblique angle included in the reconstruction and the polar angle of the frequency vector,
          the co-polar acceptance angle gamma , the sampling distance, the ring spacing,
          the alpha parameter and finally the cut-off frequency.
          The alpha and fc parameters are designed to minimize the amplification of noise.
        */
    ColsherFilter(int height_v, int width_v, float gamma_v, float theta_max_v,
		  float d_a_v, float d_b_v,
                  float alpha_colsher_axial_v, float fc_colsher_axial_v,
                  float alpha_colsher_radial_v, float fc_colsher_radial_v);


  virtual string parameter_info() const;
  
  ~ColsherFilter() {}


  private:
    //! gamma is the polar angle initialized with gamma = atan(FOV_radius/bin_size)
    float gamma;
    //! theta_max corresponds to the maximum aperture in the reconstruction with theta_max = atan(ring_spacing*(maxdelta+1)/FOV_radius)
    float theta_max;
        /* d_a, d_b are used to convert to millimeter.*/
    //! d_a is initialised with the sampling distance
    float d_a;
    //! d_b is initialised with ring spacing * sin(theta)
    float d_b; 
    //! value of the axial alpha parameter for apodized Hamming window
    float alpha_axial;
    //! value of the axial cut-off frequency of the Colsher filter
    float fc_axial;
    //! value of the planar alpha parameter for apodized Hamming window
    float alpha_planar;
    //! value of the planar cut-off frequency of the Colsher filter
    float fc_planar;
  
};


/* Utility functions (shouldn't really be declared here) TODO */

#if 0
/*!
  \brief This method filters Segment<float> data.

  Argument names encode these functions as follows:

  gamma: polar angle of the projection direction 
  theta_max: maximal aperture included in the reconstruction 
  d_a: radial distance (i.e bin_size) (in mm)
  d_b: axial sampling where d_b = ring spacing * sin(gamma) (in mm)
  fc: cut-off frequency of ramp filter
  alpha: value of the alpha parameter for Hamming window
  PadS: transaxial extension for FFT
  PadZ: axial extension for FFT
*/
void  Filter_proj_Colsher(Segment<float> &segment,			
                          float gamma, float theta_max, float d_a, float d_b,
                          float alpha_axial, float fc_axial, float alpha_planar, float fc_planar,
                          int PadS, int PadZ);

//! \brief This method is similar to the one above but with minimum and maximum ring values specified as arguments.
void  Filter_proj_Colsher(Segment<float>& segment,
                          float gamma, float theta_max, float d_a, float d_b,
                          float alpha_axial, float fc_axial, float alpha_planar, float fc_planar,
                          int PadS, int PadZ, int rmin, int rmax); 


//! \brief Similar to the ones above, except for the organisation of the input data: By view (2 viewgrams are needed) because of efficiency.
void Filter_proj_Colsher(Viewgram<float> & view_i,
			 Viewgram<float> & view_i1,
                         float gamma, float theta_max, float d_a, float d_b,
                         float alpha_axial, float fc_axial, float alpha_planar, float fc_planar,
                         int PadS, int PadZ, int rmin, int rmax);
#endif

void Filter_proj_Colsher(Viewgram<float> & view_i,
			 Viewgram<float> & view_i1,
                         ColsherFilter& CFilter, 
                         int PadS, int PadZ);
END_NAMESPACE_STIR

#endif //  __ColsherFilter_H__

