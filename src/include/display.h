// $Id$

#ifndef __DISPLAY_H__
#define __DISPLAY_H__

/* Functions to display Tensor3D and Tensor2D objects
   Version 1.0: KT

   The display functions of Tensor3D objects take parameters as follows:
   - plane_stack, or plane : 
        the Tensor object
   - scale_factors : 
        a VectorWithOffset of factors which are multiplied with the numbers
	in the Tensor object to give the "real" values
   - text :
        a VectorWithOffset of strings that are displayed below the images
   - maxi :
        a double which gives the ("real") value that will correspond to the
	maximum of the color scale. All bigger values are displayed with the
	same color. 
	If maxi is 0, all planes are scaled independently.
   - zoom :
        an int giving the number of times the image should be enlarged.
	Enlargement currently is with linear interpolation, giving
	reasonably smooth images (although one could want to see the 
	'pixels', but I didn't implement that yet).
	If zoom = 0, maximum enlargement is used.
   - num_in_window :
        an int giving the desired maximum number of images displayed in one 
	window (there can be less images if they don't fit).
	If num_in_window = 0, the maximum number of images will be displayed.

   Note that the scale_factors and text arrays are supposed to have the
   same range is the outer dimension of the Tensor3D object.

   For Tensor2D objects, the parameters are similar, but scale_factors and
   text are not vectors anymore.

   Note that there is an effective threshold at 0 currently (i.e. negative
   numbers are cut out)..
*/

/* Below is the main function to display Tensor3D objects.

   This function is templated for generality.
   NUMBER is the type of elements in the Tensor3D
   SCALE is the type of the scale factors
   CHARP is the type of the text strings (could be String as well)

   Templates instantiated in display.cxx:
   NUMBER : short, SCALE : short, CHARP : char *
   NUMBER : short, SCALE : float, CHARP : char *
   NUMBER : float, SCALE : float, CHARP : char *
   If you need other types, you'll have to recompile display.cxx
   */
template <class NUMBER, class SCALE, class CHARP>
#ifndef __MSL__
// KT 17/02/98 for some reason CodeWarrior Pro protests to the extern
extern 
#endif
void display(const Tensor3D<NUMBER>& plane_stack,
             const VectorWithOffset<SCALE>& scale_factors,
             const VectorWithOffset<CHARP>& text,
             double maxi = 0,
             int zoom = 0,
             int num_in_window = 0);



/* A version without scale factors and text */
template <class NUMBER>
inline void display(const Tensor3D<NUMBER>& plane_stack,
	     double maxi = 0, int zoom = 0,
	     int num_in_window = 0)
{
  VectorWithOffset<Real> scale_factors(plane_stack.get_min_index3(), 
				plane_stack.get_max_index3());
  scale_factors.fill(1.);
  VectorWithOffset<char *> text(plane_stack.get_min_index3(), 
				plane_stack.get_max_index3());
  text.fill("");

  display(plane_stack, scale_factors, text,
	  maxi, zoom, num_in_window);
}


/* The function for Tensor2D objects */
template <class NUMBER, class SCALE, class CHARP>
inline void display(const Tensor2D<NUMBER>& plane,
		    const SCALE scale_factor,
		    const CHARP& text,
		    double maxi = 0, int zoom = 0,
		    int num_in_window = 0)
{ 
  // KT 06/02/98 added more size parameters
  Tensor3D<NUMBER> stack(0,0,
			 plane.get_min_index2(),plane.get_max_index2(),
			 plane.get_min_index1(),plane.get_max_index1());
  stack[0] = plane;
  VectorWithOffset<SCALE> scale_factors(1);
  scale_factors[0] = scale_factor;
  VectorWithOffset<CHARP> texts(1);
  texts[0] = text;
  
  display(stack, scale_factors, texts, maxi, zoom, num_in_window);
}



/* A version without scale factors and text */
template <class NUMBER>
inline void display(const Tensor2D<NUMBER>& plane,
	     double maxi = 0, int zoom = 0,
	     int num_in_window = 0)
{
  // KT 06/02/98 added Real conversion to 0, to avoid calling a new 
  // template with SCALE=int
  display(plane, Real(0), "", maxi, zoom, num_in_window);
}

/* CL 061098 Move two function from main_promis, promis_span... to here */
#if 1
template <class NUMBER, class CHARP>
void my_display(const Tensor3D<NUMBER> &image, CHARP text); 

template <class NUMBER>
void display2D(const Tensor2D<NUMBER> &plane);


#include "sinodata.h"
void display_8_views(const PETViewgram& v1, const PETViewgram& v2, 
                     const PETViewgram& v3, const PETViewgram& v4,
                     const PETViewgram& v5, const PETViewgram& v6, 
                     const PETViewgram& v7, const PETViewgram& v8);
#endif 
#endif 
