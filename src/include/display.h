// $Id$: $Date$

#ifndef __DISPLAY_H__
#define __DISPLAY_H__

/*! 
  \file
  
  \brief   Functions to display 2D and 3D Tensor objects

  \author Kris Thielemans
  \author PARAPET project

  \date    $Date$

  \version $Revision$
*/


#include "VectorWithOffset.h"
#include "Tensor3D.h"

START_NAMESPACE_TOMO

# if defined(__GNUC__) && (__GNUC__ == 2 && __GNUC_MINOR__ >= 95)
// gcc 2.95.2 is the only compiler we've used that handles the defaults properly

/*! \brief The main function to display Tensor3D<elemT> objects.

   This function is templated for generality.
   elemT is the type of elements in the Tensor3D<elemT>
   scaleT is the type of the scale factors
   CHARP is the type of the text strings (could be char * or variations with const)

   \param plane_stack
        the Tensor object
   \param scale_factors 
        a VectorWithOffset of factors which are multiplied with the numbers
	in the Tensor object to give the "real" values
   \param text
        a VectorWithOffset of strings that are displayed below the images
   \param maxi
        a double which gives the ("real") value that will correspond to the
	maximum of the color scale. All bigger values are displayed with the
	same color. 
	If maxi is 0, all planes are scaled independently.
   \param title
        text that will be used as a title for the display. title==0 means no text.
   \param zoom 
        an int giving the number of times the image should be enlarged.
	Enlargement currently is with linear interpolation, giving
	reasonably smooth images (although one could want to see the 
	'pixels', but I didn't implement that yet).
	If zoom = 0, maximum enlargement is used.

   Note that the scale_factors and text Tensors are required to have the
   same range is the outer dimension of the Tensor3D<elemT> object.

   Note that there is an effective threshold at 0 currently (i.e. negative
   numbers are cut out)..

   \warning Due to the choice of defaults, if \c maxi is not given, the images will
   be scaled independently.

   \warning due to compiler problems, it might be that the prototypes actually
   declared are different from what you see in the auto-generated doc. 
   Check display.h
  */
// TODO, make version with CHARP ='string' but requires function overloading
template <class elemT, class scaleT, class CHARP>
void display(const Tensor3D<elemT>& plane_stack,
             const VectorWithOffset<scaleT>& scale_factors,
             const VectorWithOffset<CHARP>& text,
             double maxi = 0,
	     const char * const title = 0,
             int zoom = 0);

//! \brief display for Tensor3D<elemT> without scale factors and text.
template <class elemT>
inline void display(const Tensor3D<elemT>& plane_stack,
	     double maxi = 0,
	     const char * const title = 0, int zoom = 0);


//! \brief display for Tensor2D<elemT> objects, text==0 means no text.  
template <class elemT>
inline void display(const Tensor2D<elemT>& plane,
		    const char * const text = 0,
		    double maxi = 0, int zoom = 0);


#else // !gcc 2.95


// VC and gcc 2.8.1 have problems with the defaults in the above declarations.
// So, we have to do them by hand...

template <class elemT, class scaleT, class CHARP>
void display(const Tensor3D<elemT>& plane_stack,
             const VectorWithOffset<scaleT>& scale_factors,
             const VectorWithOffset<CHARP>& text,
             double maxi,
	     const char * const title ,
             int zoom);

template <class elemT, class scaleT, class CHARP>
inline void display(const Tensor3D<elemT>& plane_stack,
             const VectorWithOffset<scaleT>& scale_factors,
             const VectorWithOffset<CHARP>& text,
             double maxi,
	     const char * const title);

template <class elemT, class scaleT, class CHARP>
inline void display(const Tensor3D<elemT>& plane_stack,
             const VectorWithOffset<scaleT>& scale_factors,
             const VectorWithOffset<CHARP>& text,
             double maxi);


template <class elemT, class scaleT, class CHARP>
inline void display(const Tensor3D<elemT>& plane_stack,
             const VectorWithOffset<scaleT>& scale_factors,
             const VectorWithOffset<CHARP>& text);

template <class elemT>
inline void display(const Tensor3D<elemT>& plane_stack,
             double maxi,
	     const char * const title,
	     int zoom);

template <class elemT>
inline void display(const Tensor3D<elemT>& plane_stack,
             double maxi,
	     const char * const title);

template <class elemT>
inline void display(const Tensor3D<elemT>& plane_stack,
             double maxi);

template <class elemT>
inline void display(const Tensor3D<elemT>& plane_stack);

template <class elemT>
inline void display(const Tensor2D<elemT>& plane,
		    const char * const text,
		    double maxi, int zoom);

template <class elemT>
inline void display(const Tensor2D<elemT>& plane,
		    const char * const text,
		    double maxi);

template <class elemT>
inline void display(const Tensor2D<elemT>& plane,
		    const char * const text);

template <class elemT>
inline void display(const Tensor2D<elemT>& plane);

#endif


#include "sinodata.h"
void display_8_views(const PETViewgram& v1, const PETViewgram& v2, 
                     const PETViewgram& v3, const PETViewgram& v4,
                     const PETViewgram& v5, const PETViewgram& v6, 
                     const PETViewgram& v7, const PETViewgram& v8);

#include "display.inl"

END_NAMESPACE_TOMO

#endif 
