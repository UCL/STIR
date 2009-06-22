//
// $Id$
//
/*! 
  \file
  \ingroup display
  
  \brief  functions to display 2D and 3D stir::Array objects

  \author Kris Thielemans
  \author PARAPET project

  $Date$

  $Revision$

*/
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

#include "stir/IndexRange3D.h"

START_NAMESPACE_STIR
/*!
   \see display(const Array<3,elemT>&,
             const VectorWithOffset<scaleT>& ,
             const VectorWithOffset<CHARP>& ,
             double,
	     const char * const ,
             int zoom) for more info.
   This function sets the 'text' parameter to a sequence of numbers.
*/
template <class elemT>
void 
display(const Array<3,elemT>& plane_stack,
	double maxi,
	const char * const title,
	int zoom)

{
  VectorWithOffset<float> scale_factors(plane_stack.get_min_index(), 
				plane_stack.get_max_index());
  scale_factors.fill(1.);
  VectorWithOffset<char *> text(plane_stack.get_min_index(), 
				plane_stack.get_max_index());
  
  for (int i=plane_stack.get_min_index();i<= plane_stack.get_max_index();i++)
  {
    text[i]=new char[10];
    sprintf(text[i],"%d", i);
  }
  
  display(plane_stack, scale_factors, text, 
          maxi, title,zoom);
  // clean up memory afterwards
  for (int i=plane_stack.get_min_index();i<= plane_stack.get_max_index();i++)
    delete[] text[i];

}

/*! 
  \see display(const Array<3,elemT>&,
             const VectorWithOffset<scaleT>& ,
             const VectorWithOffset<CHARP>& ,
             double,
	     const char * const ,
             int zoom) for more info.*/
template <class elemT>
void 
display(const Array<2,elemT>& plane,
		    const char * const text,
		    double maxi, int zoom )
{ 
 
  if (plane.get_length()==0)
    return;
  // make a 3D array with arbitrary dimensions for its first and only plane
  Array<3,elemT> stack(IndexRange3D(0,0,0,0,0,0));
  // this assignment sets correct dimensions for the 2 lowest dimensions
  stack[0] = plane;
  VectorWithOffset<float> scale_factors(1);
  scale_factors[0] = 1.F;
  VectorWithOffset<const char*> texts(1);
  texts[0] = "";
  
  display(stack, scale_factors, texts, maxi, text, zoom);
}

# if defined(__GNUC__) && (__GNUC__ == 2 && __GNUC_MINOR__ >= 95)
// gcc 2.95.2 is the only compiler we've used that handles the defaults properly

#else 
// VC and gcc 2.8.1 have problems with the defaults in the above declarations.
// So, we have to do them by hand...


template <class elemT, class scaleT, class CHARP>
void display(const Array<3,elemT>& plane_stack,
             const VectorWithOffset<scaleT>& scale_factors,
             const VectorWithOffset<CHARP>& text,
             double maxi,
	     const char * const title)
{ display(plane_stack, scale_factors, text, maxi, title, 0); }

template <class elemT, class scaleT, class CHARP>
void display(const Array<3,elemT>& plane_stack,
             const VectorWithOffset<scaleT>& scale_factors,
             const VectorWithOffset<CHARP>& text,
             double maxi)
{ display(plane_stack, scale_factors, text, maxi, 0, 0); }


template <class elemT, class scaleT, class CHARP>
void display(const Array<3,elemT>& plane_stack,
             const VectorWithOffset<scaleT>& scale_factors,
             const VectorWithOffset<CHARP>& text)
{ display(plane_stack, scale_factors, text, 0., 0, 0); }

template <class elemT>
void display(const Array<3,elemT>& plane_stack,
	     double maxi,
	     const char * const title)
{ display(plane_stack, maxi, title, 0); }

template <class elemT>
void display(const Array<3,elemT>& plane_stack,
	     double maxi)
{ display(plane_stack, maxi, 0, 0); }

template <class elemT>
void display(const Array<3,elemT>& plane_stack)
{ display(plane_stack, 0., 0, 0); }

template <class elemT>
void display(const Array<2,elemT>& plane,
		    const char * const text,
		    double maxi)
{ display(plane, text, maxi, 0); }

template <class elemT>
void display(const Array<2,elemT>& plane,
		    const char * const text)
{ display(plane, text, 0., 0); }

template <class elemT>
void display(const Array<2,elemT>& plane)
{ display(plane, 0, 0., 0); }


#endif

END_NAMESPACE_STIR
