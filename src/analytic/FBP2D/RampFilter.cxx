//
//
/*!

  \file
  \ingroup FBP2D

  \brief Implementation of class stir::RampFilter

  \author Kris Thielemans
  \author Claire Labbe
  \author Darren Hogg
  \author PARAPET project

*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2011, Hammersmith Imanet Ltd

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

#include "stir/analytic/FBP2D/RampFilter.h"
#include <math.h>
#include <iostream>
#ifdef BOOST_NO_STRINGSTREAM
#include <strstream.h>
#else
#include <sstream>
#endif

/* Note: #ifdef NRFFT, then the Numerical Recipes version is used (if you have it...) */

START_NAMESPACE_STIR

// This function computes the samples of the ramp filter in real space
// (as opposed to frequency space)
// The formulas are complicated, but derived in Mathematica
// by computing the analytic inverse Fourier transform of a cut-off ramp
// times a Hamming window.

static inline float 
ramp_filter_in_space(const int n, 
			   const float sampledist, 
			   const int length, 
			   const float alpha, 
			   const float fc)
{
  const double x = n*2*fc;
  // KT&Darren Hogg 17/05/2000 removed square(sampledist) as this introduced a scaling factor in the reconstructions
  if (n==0)
    return
      static_cast<float>((2*square(fc)*(-4 + alpha*(4 + square(_PI))))/(_PI/* *square(sampledist) */));
  else if (fabs(fabs(x)-1) < 1E-6)
    return
        static_cast<float>(
			   -(square(2*fc)*(8*alpha + (-1 + alpha)*square(_PI)))/
			   (4*_PI/* *square(sampledist) */));
  else
    return
      static_cast<float>(
        square(2*fc)*(
	   -alpha - square(x) + 3*alpha*square(x) - square(square(x)) + 
           _PI*x*sin(_PI*x)*(-1 + square(x))*
                  (-alpha + (-1 + 2*alpha)*square(x)) + 
           cos(_PI*x)*(alpha - (1 + alpha)*square(x) + 
                      (-1 + 2*alpha)*square(square(x)))
        )/
        (_PI/* *square(sampledist) */*square(-1 + x)*square(x)*square(1 + x))
	);
}

// KT&CL 03/08/99 insert max value for fc
RampFilter::RampFilter(float sampledist_v, int length , float alpha_v, float fc_v)
  :
#ifdef NRFFT
  Filter1D <float>(length), 
#endif
  fc(min(fc_v, .5F)), alpha(alpha_v), sampledist(sampledist_v)
{

  start_timers();

  // Necessary exit for the silly case when length==0
  if (length==0) 
    return;
#ifdef OLDRAMP
  /* This is the straightforward implementation:
     define it in complex space as abs(frequency).
     This has a well-known problem that DC values are wrong. This is essentially because
     the ramp filter is a continuous filter. Discrete convolution should be done
     with the samples of the continuous fourier transform of the ramp.
  */
  // KT&DH 17/05/2000 TODO: highly suspect that the scale factor is inappropriate
#error check scale factor in ramp filter!
/* As realft uses only positive frequencies, the filter needs to be defined
   only for those frequencies, so it has length/2 elements. 
   However, in general the values are complex, so the numbers of 
   real numbers is 2*length/2==length.
   */
  float           f = 0.0;

  for (Int i = 1; i <= length - 1; i += 2) {
    f = (float) ((float) 0.5 * (i - 1) / length);
    float nu_a = f ;
    if (f <= fc)
      filter[i] = nu_a * (alpha + (1. - alpha) * cos(_PI * f / fc));
    else
      filter[i] = 0.0;
    filter[i + 1] = 0.0;	/* imaginary part */
  }
  if (0.5 <= fc)		/* see realft for an explanation:data[2]=last real */
    filter[2] = (0.5) * (alpha + (1. - alpha) * cos(_PI * f / fc));
  else
    filter[2] = 0.;
#else
  /* This computes the ramp filter in frequency space in 2 steps:
     - sample the filter in real space
     - perform a discrete FT to find values in the frequency domain
     This gives better agreement with the filtering of a band-limited
     function with the analytic ramp (with cut-off).
     In particular, it solves a problem with the DC component of the 
     filter. Sampling the ramp in the frequency domain gives 0 for 
     DC component, resulting in images with negative tails.
     */

  assert(length%2==0);

  // first construct filter in 'real' space
#ifdef NRFFT
  filter.set_offset(0);
#else
  Array<1,float> filter(length);
#endif

  filter[0] = ramp_filter_in_space(0, sampledist, length, alpha, fc);
  // note: filter[length/2] is set twice for even length, but that's fine
  for (int n = 1; n <= length/2; n += 1)
  {
    filter[n] = ramp_filter_in_space(n, sampledist, length, alpha, fc);
    filter[length-n] = filter[n];
  }

  //std::cerr <<"Ramp filter in real space = " <<filter;
#ifdef NRFFT
  filter.set_offset(1);

  realft(filter, length/2, 1);

  //std::cerr <<"Ramp filter in Fourier space = "<<filter;
#else
  if (set_kernel(filter) == Succeeded::no)
    error("Error initialisation ramp filter\n");

  //std::cerr <<"Ramp filter in Fourier space = " <<complex_kernel;
#endif
#endif
  stop_timers();


};


std::string RampFilter:: parameter_info() const
{
#ifdef BOOST_NO_STRINGSTREAM
  // dangerous for out-of-range, but 'old-style' ostrstream seems to need this
  char str[1000];
  ostrstream s(str, 1000);
#else
  std::ostringstream s;
#endif  
    s << "RampFilter :="
      << "\nFilter length := "
#ifdef NRFFT
      << filter.get_length()
#else
      << (kernel_in_frequency_space.size()-1)*2
#endif
      << "\nCut-off in cycles := "<< fc
      << "\nAlpha parameter := "<<alpha
      << "\nSample dist := "<< sampledist;
    
   return s.str();
}



END_NAMESPACE_STIR
