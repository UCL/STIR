//
// $Id$: $Date$
//
/*! 
 \file

  
  \brief  inline functions to display 2D and 3D Tensor objects

  \author Kris Thielemans
  \author PARAPET project

  \date    $Date$

  \version $Revision$

*/
/*!
   \see display(const Tensor3D<elemT>&,
             const VectorWithOffset<scaleT>& ,
             const VectorWithOffset<CHARP>& ,
             double,
	     const char * const ,
             int zoom) for more info.
   This function sets the 'text' parameter to a sequence of numbers.
*/
template <class elemT>
void 
display(const Tensor3D<elemT>& plane_stack,
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
  \see display(const Tensor3D<elemT>&,
             const VectorWithOffset<scaleT>& ,
             const VectorWithOffset<CHARP>& ,
             double,
	     const char * const ,
             int zoom) for more info.*/
template <class elemT>
void 
display(const Tensor2D<elemT>& plane,
		    const char * const text,
		    double maxi, int zoom )
{ 
 
  if (plane.get_length()==0)
    return;
  // make a 3D array with arbitrary dimensions for its first and only plane
  Tensor3D<elemT> stack(0,0,0,0,0,0);
  // this assignment sets correct dimensions for the 2 lowest dimensions
  stack[0] = plane;
  VectorWithOffset<float> scale_factors(1);
  scale_factors[0] = 1.F;
  VectorWithOffset<const char*> texts(1);
  texts[0] = text==0 ? "" : text;
  
  display(stack, scale_factors, texts, maxi, 0/*title*/, zoom);
}

# if defined(__GNUC__) && (__GNUC__ == 2 && __GNUC_MINOR__ >= 95)
// gcc 2.95.2 is the only compiler we've used that handles the defaults properly

#else 
// VC and gcc 2.8.1 have problems with the defaults in the above declarations.
// So, we have to do them by hand...


template <class elemT, class scaleT, class CHARP>
void display(const Tensor3D<elemT>& plane_stack,
             const VectorWithOffset<scaleT>& scale_factors,
             const VectorWithOffset<CHARP>& text,
             double maxi,
	     const char * const title)
{ display(plane_stack, scale_factors, text, maxi, title, 0); }

template <class elemT, class scaleT, class CHARP>
void display(const Tensor3D<elemT>& plane_stack,
             const VectorWithOffset<scaleT>& scale_factors,
             const VectorWithOffset<CHARP>& text,
             double maxi)
{ display(plane_stack, scale_factors, text, maxi, 0, 0); }


template <class elemT, class scaleT, class CHARP>
void display(const Tensor3D<elemT>& plane_stack,
             const VectorWithOffset<scaleT>& scale_factors,
             const VectorWithOffset<CHARP>& text)
{ display(plane_stack, scale_factors, text, 0., 0, 0); }

template <class elemT>
void display(const Tensor3D<elemT>& plane_stack,
	     double maxi,
	     const char * const title)
{ display(plane_stack, maxi, title, 0); }

template <class elemT>
void display(const Tensor3D<elemT>& plane_stack,
	     double maxi)
{ display(plane_stack, maxi, 0, 0); }

template <class elemT>
void display(const Tensor3D<elemT>& plane_stack)
{ display(plane_stack, 0., 0, 0); }

template <class elemT>
void display(const Tensor2D<elemT>& plane,
		    const char * const text,
		    double maxi)
{ display(plane, text, maxi, 0); }

template <class elemT>
void display(const Tensor2D<elemT>& plane,
		    const char * const text)
{ display(plane, text, 0., 0); }

template <class elemT>
void display(const Tensor2D<elemT>& plane)
{ display(plane, 0, 0., 0); }

#endif
