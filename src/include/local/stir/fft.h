//
// $Id$
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

  $Date :$

  $Revision$
*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/
#ifndef __FFT_H__
#define __FFT_H__

#include "stir/common.h"

START_NAMESPACE_STIR

template <int num_dimensions, typename elemT> class Array;

void four1(Array<1,float> &data, int nn, int isign);
void fourn (Array<1,float> &data, Array<1,int> &nn, int ndim, int isign);
//void convlv (Array<1,float> &data, const Array<1,float> &filter, int n);
void convlvC (Array<1,float> &data, const Array<1,float> &filter, int n);
//void rlft3(Array<3,float> &data, Array<2,float> &speq, unsigned long nn1, unsigned long nn2, unsigned long nn3, int isign);
void realft ( Array<1,float> &data, int n, int isign);

END_NAMESPACE_STIR

#endif

