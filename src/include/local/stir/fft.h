//
//
//
/*!

  \file
  \brief Declaration of FFT routines

  This are C++-ified version of Numerical Recipes routines. You need to have
  the book before you can use this!

  \warning This will be removed.

  \author Claire Labbe
  \author Kris Thielemans
  \author PARAPET project


*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2004, Hammersmith Imanet
    See STIR/LICENSE.txt for details
*/
#ifndef __FFT_H__
#define __FFT_H__

#include "stir/common.h"

START_NAMESPACE_STIR

template <int num_dimensions, typename elemT> class Array;

//! 1-dimensional FFT
/*!
  Replaces data by its discrete Fourier transform, if isign is input
  as 1; or replaces data by nn times its inverse discrete Fourier transform, 
  if isign is input as -1. data is a complex array of length nn, input as a 
  real array data[1..2*nn]. nn MUST be an integer power of 2 ( this is not   
  checked for !).
*/

void four1(Array<1,float> &data, int nn, int isign);
//! n-dimensional FFT
/*!
  Replaces data by its ndim-dimensional discrete Fourier transform, 
  if isign is input  as  1. nn[1..ndim] is an integer array containing the  
  lengths of each dimension (number of complex values), which MUST all
  be powers of 2. data is a real array of length twice the product of
  thes lengths, in which the data are stored as in a multidimensional 
  complex array: real and imaginary parts of each  element are in 
  consecutive locations , and the rightmost index of the array 
  increases most rapidly as one  proceeds along data. For a 
  two-dimensional array, this  is equivalent  to storing  the 
  array  by rows . If isign  is in input as -1 , data is replaced by 
  its inverse transform time the product of the lengths of all dimensions.
*/
void fourn (Array<1,float> &data, Array<1,int> &nn, int ndim, int isign);
//void convlv (Array<1,float> &data, const Array<1,float> &filter, int n);
//! Convolve data with a filter which is given in frequency space
/*! \param data has to have a min_index == 1
    \param filter has to in the format produced by realft. 
    \param n has to be equal to th elength of \a data

  \warning Currently, only the real elements of the filter will be used. (Historical reason?)
  So, the result will be incorrect of the filter has complex components (i.e. its 
  spatial kernel is not symmetric).
  */
void convlvC (Array<1,float> &data, const Array<1,float> &filter, int n);

//! 3D FFT of real numbers
void rlft3(Array<3,float> &data, Array<2,float> &speq, int nn1, int nn2, int nn3, int isign);
//! Calculates the Fourier Transform of a set of 2n real-valued data points.
/*!
  Replaces this data ( which is stored in array data[1..2n]) by the positive
  frequency half of its complex Fourier Transform. The real-valued first and
  last components of the complex transform are returned as element data[1] and
  data[2] respectively. n must be a power of 2. The routine also calculates
  the inverse transform of a complex data array if it is the transform of real
  data. (Result in this case must be multiplied by 1/n.)
*/

void realft ( Array<1,float> &data, int n, int isign);

END_NAMESPACE_STIR

#endif

