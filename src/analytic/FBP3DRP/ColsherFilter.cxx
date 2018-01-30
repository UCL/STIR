//
//

/*! 
  \file 
  \ingroup FBP3DRP
  \brief Colsher filter implementation
  \author Kris Thielemans
  \author Claire LABBE
  \author based on C-code by Matthias Egger
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

#ifdef NRFFT
#include "stir/Viewgram.h"
#endif
#include "stir/analytic/FBP3DRP/ColsherFilter.h"
#include "stir/numerics/fourier.h"
#include "stir/IndexRange2D.h"
#include "stir/Succeeded.h"

#ifdef __DEBUG_COLSHER
// for debugging...
#include "stir/display.h"
#include "stir/IO/write_data.h"
#include <iostream>
#include <fstream>
#endif

#include <math.h>
#ifdef BOOST_NO_STRINGSTREAM
#include <strstream.h>
#else
#include <sstream>
#endif

START_NAMESPACE_STIR

std::string ColsherFilter::parameter_info() const
{
#ifdef BOOST_NO_STRINGSTREAM
  // dangerous for out-of-range, but 'old-style' ostrstream seems to need this
  char str[2000];
  ostrstream s(str, 2000);
#else
  std::ostringstream s;
#endif  
    s << "\nColsherFilter Parameters :="
#ifdef NRFFT
      << "\nFilter height := "<< height
      << "\nFilter width := "<< width      
#endif
      << "\nMaximum aperture (theta_max) := "<< theta_max
      << "\nCut-off in cycles along planar direction:= "<< fc_planar
      << "\nCut-off in cycles along axial direction:= "<< fc_axial
      << "\nAlpha parameter along planar direction := "<<alpha_planar
      << "\nAlpha parameter along axial direction:= "<<alpha_axial
#ifdef NRFFT
      << "\nRadial sampling (along bin direction) := "<< d_a
      << "\nAxial sampling (along ring direction) := "<< d_b
      << "\nCopolar angle (gamma):= "<< gamma
#endif
      ;
        
    return s.str();
}

#ifndef NRFFT



ColsherFilter::ColsherFilter(float theta_max_v,
			     float alpha_colsher_axial_v, float fc_colsher_axial_v,
			     float alpha_colsher_planar_v, float fc_colsher_planar_v,
			     const int stretch_factor_axial,
			     const int stretch_factor_planar) 
        : theta_max(theta_max_v), 
          alpha_axial(alpha_colsher_axial_v), fc_axial(fc_colsher_axial_v),
          alpha_planar(alpha_colsher_planar_v), fc_planar(fc_colsher_planar_v),
	  stretch_factor_axial(stretch_factor_axial),
	  stretch_factor_planar(stretch_factor_planar)
{
}


#ifdef __DEBUG_COLSHER
#ifndef NRFFT
//a function to get the real part of a complex number used in the debugging stuff
float real_part(const std::complex<float>& z) { return z.real(); }
#endif
#endif
 
Succeeded
ColsherFilter::
set_up(int target_height, int target_width, float theta,
       float d_a, float d_b)
{
  /* Ideally we would sample the Colsher filter in 'real' space, and then
     construct the coefficients for discrete fourier transforms. Unfortunately, 
     this requires an impossible (?) analytic integral.
     We approximate this integral numerically y using the following procedure.

     We first define the Colsher filter on a larger grid in 
     frequency space. Then we inverse transform to 'real' space,
     keep only the filter coefficients for the actual grid (i.e. cut
     off the tails), and transform back to frequency space.
  */
  const int height=target_height*stretch_factor_axial;
  const int width=target_width*stretch_factor_planar;
  if (height==0 || width==0)
    return Succeeded::yes;
  if (theta > theta_max+.001F)
    {
      warning("ColsherFilter::set_up called with theta %g larger than theta_max %g",
	      theta, theta_max);
      return Succeeded::no;
    }
  start_timers();

  //*********** first construct filter on large grid 
  /*
    The Colsher filter is real-valued and symmetric. As we use 
    fourier_for_real_data, we have to arrange it in wrap-around order in the 
    axial dimension, but we need only the positive frequencies in 
    tangential direction.
   */
  Array<2,std::complex<float> > filter(IndexRange2D(height,width/2+1));

  // KT&Darren Hogg 03/07/2001 inserted correct scale factor 
  // TODO this assumes current value for the magic_number in backprojector
  const float scale_factor = static_cast<float>(4*_PI*d_a);
 
  for (int j = 0; j <=  height/2; ++j) 
    {
      const float fb = static_cast<float>(j) / height;
      const float nu_b = fb / d_b;
      for (int k = 0; k <= width/2; ++k) 
	{
	  const float fa = (float) k / width;
	  const float nu_a = fa / d_a;

	  float fil = 0;
	  if (fa < fc_planar && fb < fc_axial) 
	    {
	      /* Colsher filter */
	      const float omega = atan2(nu_b, nu_a);
	      const float psi = acos(sin(omega) * cos(theta));
	      const float mod_nu = sqrt(nu_a * nu_a + nu_b * nu_b);
		
	      if (cos(psi) >= cos(theta_max))
		fil = static_cast<float>(mod_nu / 2. / _PI);
	      else
		fil = static_cast<float>(mod_nu / 4. / asin(sin(theta_max) / sin(psi)));
	      /* Apodizing Hanning window */;
	      fil *= 
		static_cast<float>(
				   (alpha_planar + (1. - alpha_planar) * cos(_PI * fa / fc_planar))
				   *(alpha_axial + (1. - alpha_axial)* cos(_PI * fb / fc_axial)));
	      fil *= scale_factor;
	    }
	  filter[j][k] = fil;
	  if (j>0)
	    filter[height-j][k] = fil;
	}
    }
  //*********** now find it on normal grid by passing to 'real' space

  if (stretch_factor_planar>1 || stretch_factor_axial>1)
    {
      Array<2,float > colsher_real=
	inverse_fourier_for_real_data_corrupting_input(filter);
      // cut out tails. unfortunately that's a bit complicated because of wrap-around
      colsher_real.resize(IndexRange2D(target_height,target_width));
      for (int j = 0; j <  target_height/2; ++j) 
	{
	  for (int k = 0; k < target_width/2; k++) 
	    {
	      if (j!=0) colsher_real[target_height-j][k] = colsher_real[j][k];
	      if (j!=0 &&k!=0) colsher_real[target_height-j][target_width-k] = colsher_real[j][k];
	      if (k!=0) colsher_real[j][target_width-k] = colsher_real[j][k];
	    }
	}
      filter = fourier_for_real_data(colsher_real);      
    }
  //*********** set kernel used for filtering
  const Succeeded success= set_kernel_in_frequency_space(filter);
  stop_timers();

#ifdef __DEBUG_COLSHER
  {
    // write to file
    /* a bit complicated because we can only write Array's of real numbers.
       Luckily, the Colsher filter is real in frequency space, so we can
       copy its real part into an Array of floats.
    */
    Array<2,float > real_filter(IndexRange2D(target_height,target_width/2+1));
    std::transform(filter.begin_all(), filter.end_all(), real_filter.begin_all(), 
		   real_part/*std::real<std::complex<float> >*/);
    char file[200];
    sprintf(file,"%s_%d_%d_%g.dat","new_colsher",target_width,target_height,theta);
    std::cout << "Saving filter : " << file << std::endl;
    std::ofstream s(file);
    write_data(s,real_filter);
  }
#endif
#ifdef __DEBUG_COLSHER
  {
    Array<2,float > PSF(IndexRange2D(-0,target_height/3,-target_width/3,target_width/3));
    PSF.fill(0);
    PSF[0][0]=1;
    this->operator()(PSF);
    display(PSF,"PSF",PSF.find_max()/20);
  }
#endif
  return success;
}
                
#else //NRFFT

ColsherFilter::ColsherFilter(int height_v, int width_v, float gamma_v, float theta_max_v,
				 float d_a_v, float d_b_v,
                                 float alpha_colsher_axial_v, float fc_colsher_axial_v,
                                 float alpha_colsher_planar_v, float fc_colsher_planar_v) 
        : Filter2D<float>(height_v, width_v),gamma(gamma_v), theta_max(theta_max_v), d_a(d_a_v), d_b(d_b_v),
          alpha_axial(alpha_colsher_axial_v), fc_axial(fc_colsher_axial_v),
          alpha_planar(alpha_colsher_planar_v), fc_planar(fc_colsher_planar_v)
{
 
  if (height==0 || width==0)
    return;

    int             k, j;
    float           fa, fb, omega, psi;
    float           mod_nu, nu_a, nu_b;
    double fil;
	/*
	 * The Colsher filter is real-valued, so it has only height*width elements,
	 * going from [1..height*width]. It is arranged in wrap-around order
	 * in both dimensions, see Num.Rec.C, page 523
	 */
 
    int ii = 1;//float fmax = 0.F;
	
    // TODO don't use 0.5* for upper boundary
    for (j = 0; j <= 0.5 * height; j++) {
        fb = (float) j / height;
        nu_b = fb / d_b;
        for (k = 0; k <= 0.5 * width; k++) {
            fa = (float) k / width;
            nu_a = fa / d_a;
            if (fa == 0. && fb == 0.) {
                filter[ii++] = 0.F;
                continue;
            }

            omega = atan2(nu_b, nu_a);
            psi = acos(sin(omega) * sin(gamma));
            mod_nu = sqrt(nu_a * nu_a + nu_b * nu_b);

            /* Colsher formula = fil = mod_nu / 4. / asin(sin(theta_max) / sin(psi)); */
            if (cos(psi) >= cos(theta_max))
                fil = mod_nu / 2. / _PI;
            else
                fil = mod_nu / 4. / asin(sin(theta_max) / sin(psi));
                      
                       
                /* Apodizing Hanning window */;
                //CL 250899 In order to make similar to the CTI program, we use a damping window
                //for both planar and axial direction

            if (fa < fc_planar && fb < fc_axial) 
                filter[ii++] = 
		  static_cast<float>(
				     fil * (alpha_planar + (1. - alpha_planar) * cos(_PI * fa / fc_planar))
				     *(alpha_axial + (1. - alpha_axial)* cos(_PI * fb / fc_axial)));
            else
                filter[ii++] = 0.F;

        }
                
               
        for (k = int (-0.5 * width) + 1; k <= -1; k++) {
            fa = (float) k / width;
            nu_a = fa / d_a;
            omega = atan2(nu_b, -nu_a);
            psi = acos(sin(omega) * sin(gamma));
               
                        
            mod_nu = sqrt(nu_a * nu_a + nu_b * nu_b);

            if (cos(psi) >= cos(theta_max))
                fil = mod_nu / 2. / _PI;
            else
                fil = mod_nu / 4. / asin(sin(theta_max) / sin(psi));


            if (-fa < fc_planar && fb < fc_axial) 
                filter[ii++] = 
		  static_cast<float>(fil * (alpha_planar + (1. - alpha_planar) * cos(_PI * (-fa) / fc_planar))
				     *(alpha_axial + (1. - alpha_axial)* cos(_PI * fb / fc_axial)));
            else
                filter[ii++] = 0.F;
		
        }
    }
        
    for (j = int (-0.5 * height) + 1; j <= -1; j++) {
        fb = (float) j / height;
        nu_b = fb / d_b;
        for (k = 0; k <= 0.5 * width; k++) {
            fa = (float) k / width;
            nu_a = fa / d_a;
            omega = atan2(-nu_b, nu_a);
            psi = acos(sin(omega) * sin(gamma));
              
            mod_nu = sqrt(nu_a * nu_a + nu_b * nu_b);

            if (cos(psi) >= cos(theta_max))
                fil = mod_nu / 2. / _PI;
            else
                fil = mod_nu / 4. / asin(sin(theta_max) / sin(psi));

 
            if (fa < fc_planar && -fb < fc_axial) 
                filter[ii++] = 
		  static_cast<float>(fil * (alpha_planar + (1. - alpha_planar) * cos(_PI * fa / fc_planar))
				     *(alpha_axial + (1. - alpha_axial)* cos(_PI * (-fb) / fc_axial)));
            else
                filter[ii++] = 0.F;
                    
        }
                
        for (k = int (-0.5 * width) + 1; k <= -1; k++) {
            fa = (float) k / width;
            nu_a = fa / d_a;
            omega = atan2(-nu_b, -nu_a);
            psi = acos(sin(omega) * sin(gamma));
              
            mod_nu = sqrt(nu_a * nu_a + nu_b * nu_b);

            if (cos(psi) >= cos(theta_max))
                fil = mod_nu / 2. / _PI;
            else
                fil = mod_nu / 4. / asin(sin(theta_max) / sin(psi));

            if (-fa < fc_planar && -fb < fc_axial) 
                filter[ii++] = 
		  static_cast<float>(fil * (alpha_planar + (1. - alpha_planar) * cos(_PI * (-fa) / fc_planar))
				      *(alpha_axial + (1. - alpha_axial)* cos(_PI * (-fb) / fc_axial)));
            else
                filter[ii++] = 0.F;

                         
        }
    }

    // KT&Darren Hogg 03/07/2001 inserted correct scale factor 
    // TODO this assumes current value for the magic_number in backprojector
    filter *= static_cast<float>(4*_PI*d_a);

    
#ifdef __DEBUG_COLSHER
  {
    char file[200];
    sprintf(file,"%s_%d_%d_%g.dat","old_colsher",width,height,_PI/2-gamma);
    std::cout << "Saving filter : " << file << std::endl;
    std::ofstream s(file);
    write_data(s,filter);
  }
#endif
}

void Filter_proj_Colsher(Viewgram<float> & view_i,
			 Viewgram<float> & view_i1,
                         ColsherFilter& CFilter, 
                         int PadS, int PadZ)
{  
  // start_timers();
  
  const int rmin = view_i.get_min_axial_pos_num();
  const int rmax = view_i.get_max_axial_pos_num();
  int nrings = rmax - rmin + 1; 
  int nprojs = view_i.get_num_tangential_poss();
  
  int width = (int) pow(2, ((int) ceil(log((PadS + 1.) * nprojs) / log(2.))));
  int height = (int) pow(2, ((int) ceil(log((PadZ + 1.) * nrings) / log(2.))));	
  
  const int maxproj = view_i.get_max_tangential_pos_num();
  const int minproj = view_i.get_min_tangential_pos_num();
  
  int roffset = -rmin * width *2; 
  Array<1,float> data(1,2 * height * width);
  
  
  for (int j = rmin; j <= rmax; j++) 
  {
    for (int k = minproj; k <= maxproj; k++) {
      data[roffset + 2 * j * width + 2 * (k - minproj) + 1] = view_i[j][k];
      data[roffset + 2 * j * width + 2 * (k - minproj) + 2] = view_i1[j][k];
    }
  }
  {
    
    CFilter.apply(data);
  }
  
  
  
  
  for (int j = rmin; j <= rmax; j++) 
  {
    for (int k = minproj; k <= maxproj; k++) 
    {
      view_i[j][k] = data[roffset + 2 * j * width + 2 * (k - minproj) + 1];
      view_i1[j][k] =data[roffset + 2 * j * width + 2 * (k - minproj) + 2];
    }
  }
  
  // stop_timers();
  
}

#endif //NRFFT
END_NAMESPACE_STIR
