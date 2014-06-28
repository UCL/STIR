/*
    Copyright (C) 2000- 2007, Hammersmith Imanet Ltd
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

/*!
 \file 
 \ingroup display
  
 \brief  Functions to display stir::Array objects (2d and 3d) and stir::RelatedViewgrams
  
 \author Kris Thielemans
 \author PARAPET project
 
  

 \see display.h for some comments on the interface.

 Three implementations now, depending on which preprocessor symbols are defined
 at compile-time:

<ul>
 <li> STIR_SIMPLE_BITMAPS is based on some functions KT wrote in 1991,
   which work in XWindows, DOS (upto SVGA resolution, or using a PGA) and 
   a VAX using a Matrox card (very much similar to PGA).
   It's fairly simplistic. No menus. 

 <li> STIR_MATHLINK puts the imageinfo over a MathLink connection to Mathematica,
   where it can be displayed anyway you like.

 <li> STIR_PGM puts all images in a single PGM file.

</ul>

 if both STIR_SIMPLE_BITMAPS and STIR_MATHLINK are defined, stir::display
 asks which version you want to use
 
*/
// further doxygen comments. Only enabled when running doxygen
#ifdef DOXYGEN_SKIP
// we need to define these to get doxygen to process their comments
#define STIR_SIMPLE_BITMAPS
#define STIR_MATHLINK
#define STIR_PGM
#endif
/*! \def STIR_SIMPLE_BITMAPS
    \brief Preprocessor symbol that needs to be defined to enable X-windows 
    functionality for stir::display.

    \see display_array.cxx for some info
*/
/*! \def STIR_MATHLINK
    \brief Preprocessor symbol that needs to be defined to enable MathLink
    functionality for stir::display.

    \see display_array.cxx for some info
*/
/*! \def STIR_SIMPLE_BITMAPS
    \brief Preprocessor symbol that needs to be defined to enable PGM 
    functionality for stir::display.

    \see display_array.cxx for some info
*/

#include "stir/display.h"
#include "stir/IndexRange3D.h"
#include "stir/utilities.h"
#include "stir/RelatedViewgrams.h"
#include <iostream>

  
// First we define the different implementations. 
// See end of file for display() itself.

#ifdef STIR_SIMPLE_BITMAPS

// #include "gen.h"
// gen.h defined Min (which is used in screen.h)
#define Min std::min
#include "screen.h"
#include <cstring>

START_NAMESPACE_STIR

// local helper routine, defined after display()
template <class elemT>
static void Array2DtoSCImg (
		      image_t image[],
		      const Array<2,elemT>& plane, 
		      int scale, double maxi);

/* KT
   Warning: g++ 2.7.2.2 compiler bug:
   when using VectorWithOffset<float> for the scale_factors (and not the template 
   SCALE), g++ complains about a null character. Removing any of the other
   parameters, or using int or double as the type, makes this g++ bug 
   disappear. Or indeed, using a template...
   */
template <class elemT, class SCALE, class CHARP>
void display_bitmap(const Array<3,elemT>& plane_stack,
	     const VectorWithOffset<SCALE>& scale_factors,
	     const VectorWithOffset<CHARP>& text,
	     double maxi, const char * const title, int scale)
{
  if (plane_stack.get_length() == 0)
    return;

  Coordinate3D<int> min_indices;
  Coordinate3D<int> max_indices;

  if (!plane_stack.get_regular_range(min_indices, max_indices))
  {
    warning("display_bitmap: can only display 'regular' arrays. Returning.\n");
    return;
  }

  const int length_y = max_indices[2] - min_indices[2] + 1;
  const int length_x = max_indices[3] - min_indices[3] + 1;

  // KT 30/05/2002 open a window here first, such that SC_X_MAX is set to actual window size
  SC_START_BIG(); // KT 30/05/2002 select bigger window size 
  // window dimensions
  const int min_x = 0;
  const int min_y = 0;
  const int max_x = SC_X_MAX - 40;
  const int max_y = SC_Y_MAX;

  // first try to get all images in one window
  int num_in_window = plane_stack.get_length();

  screen_image_t *sc_image = new screen_image_t[num_in_window];
  num_in_window = center_sc_images(&scale,min_x,max_x,min_y,max_y, 
                           length_x,
			   length_y,
			   sc_image,num_in_window);
  if (num_in_window == 0)
  {
    SC_STOP();
    return;
  }

  int nr=plane_stack.get_min_index();
  int i;
  while (nr<=plane_stack.get_max_index())
  {
    if (nr!=plane_stack.get_min_index())
      SC_START_BIG(); // KT 30/05/2002 select bigger window size 
    SC_MASK(SC_M_ALL);
    SC_CLEAR_BLOCK((SC_C_BACKGROUND+ SC_C_MAX)/3,0,(int)SC_X_MAX ,0,(int)SC_Y_MAX );
    SC_SCALE(SC_X_MAX-30,30, 20, SC_Y_MAX - 60);
    // output title, making some attempt to centre it
    if (title != 0)      
      put_textstr(SC_X_MAX/2 - strlen(title)*4, 35, title);

    const long image_sizeX = scale*(length_x-1)+1;
    const long image_sizeY = scale*(length_y-1)+1;

    for(i=0; i<num_in_window && nr<=plane_stack.get_max_index(); i++, nr++)
    {       
      double thismaxi;
      
      if (maxi!=0.0)
      {
	if (scale_factors[nr] != 0) 
	  thismaxi = maxi / scale_factors[nr];
	else
	{
	  if (plane_stack[nr].find_max() == 0)
	    thismaxi = 0;
	  else
	  {
	    warning("display: Relative scale of image %d is zero", nr);
	    return;
	  }
	}
      }
      else
	thismaxi = (double)plane_stack[nr].find_max();
      
      // TODO ANCI C++ 'new' throws exception when it cannot allocate
      if ( (sc_image[i].image= new image_t[image_sizeX * image_sizeY])==0 )
      {
	warning("display: Error allocating space for image buffer. Exiting");
	return;
      }
      
      Array2DtoSCImg(sc_image[i].image,plane_stack[nr], scale,thismaxi);
      // KT 29/2/2000 
      // work-around for the fact that the (old) display library
      // does not use const char*. We copy the data into char *...
      // There would be problems if sc_image[i].text is going 
      // to be modified. However, draw_sc_images does not do this, so let's
      // live with it (otherwise it requires changing lots of declarations
      // in screen.c, and possibly X !)
      // TODO ?
      sc_image[i].text = new char[strlen((text[nr] == 0) ? "" : text[nr])];
      strcpy(sc_image[i].text, (text[nr] == 0) ? "" : text[nr]);
    }

    draw_sc_images(image_sizeX,
		   image_sizeY,
		   sc_image,i);
    SC_STOP();
    for ( i--; i>=0; i--)
    {
      //delete[] (sc_image[i].image); now deleted by XDestroyImage
      delete[] (sc_image[i].text);
    }
    if (plane_stack.get_max_index()>nr)
      if( !ask("Continue display?",true) )
	break;                          /* out of while                 */
  }
  delete[] sc_image;
}


// local functions

template <class elemT>
static void Array2DtoSCImg (
		      image_t image[],
		      const Array<2,elemT>& plane, 
		      int scale, double maxi)
{
  image_t   *pimage;
  register  image_t *pix;
  register  int y_count,x_count,i;

  // length x,y of original image
  const int org_length_y = plane.get_length();
  const int org_length_x = plane[plane.get_min_index()].get_length();
  //   length_x,y sizes of constructed image
  const int length_x = scale*(org_length_x-1)+1;
  // const int length_y = scale*(org_length_y-1)+1;

  /* pix  : address in image of current pixel */

  pimage = image; 
  for (y_count=plane.get_min_index(); y_count<=plane.get_max_index(); y_count++, pimage+=length_x*scale)
    for (x_count=plane[y_count].get_min_index(), pix=pimage; x_count<=plane[y_count].get_max_index(); x_count++, pix+=scale)
    {
      const elemT current = plane[y_count][x_count];
      if (current<=0)
        *pix = (SC_pixel_t)SC_C_BACKGROUND;
      else if ((double)current>=maxi)
        *pix = (SC_pixel_t)SC_C_MAX;
      else
        *pix = (SC_pixel_t)(SC_C_BACKGROUND +
			    (current * long(SC_C_MAX-SC_C_BACKGROUND)) / maxi);
    }
  if (scale==1) return;

  /* interpolate horizontal lines */
  pimage = image;
  for (y_count=org_length_y; y_count>0; y_count--, pimage+=length_x*scale)
    for (x_count=org_length_x-1,pix=pimage; x_count>0; x_count--,pix+=scale)
    {
/* Original, slow but good and easy
      float tmp;

      tmp=((float)*(pix+scale) - *pix)/scale ;
      for (i=1; i<scale; i++)
	*(pix + i) = (SC_pixel_t)(*pix + i*tmp + .5);
   New version, assume long is at least 8-bit longer then SC_pixel_t
   Do computations with fixed point representation of float's :
   Multiply all numbers with 0x100 (-> the rounding factor .5 becomes 0x80)
   Note: the (int) conversion in the tmp=.... line shouldn't be necessary,
         but MsC 5.1 wrongly assumes unsigned long's 
         when SC_pixel_t is unsigned.
*/
      long tmp, init;

      tmp = (((long)*(pix+scale) - (int)*pix) * 0x100) / scale ;
      for (i=1, init=*pix * 0x100L + 0x80; i<scale; i++)
        *(pix+i) = (SC_pixel_t)((init+ i*tmp) / 0x100);
    }

  /* interpolate vertical lines */
  pimage = image;
  for (y_count=org_length_y-1; y_count>0; y_count--, pimage += length_x*scale)
    for (x_count=length_x, pix=pimage; x_count>0; x_count--,pix++)
    {
/* Original
      float tmp;

      tmp=((float)*(pix + length_x*scale) - *pix)/scale ;
      for (i=1; i<scale; i++)
	*(pix + length_x*i) = (SC_pixel_t)(*pix + i*tmp + .5);
   New version
*/
      long tmp, init;

      tmp = (((long)*(pix + length_x*scale) - (int)*pix) * 0x100) / scale ;
      for (i=1, init=*pix * 0x100L + 0x80; i<scale; i++)
        *(pix + length_x*i) = (SC_pixel_t)((init+ i*tmp) / 0x100);
    }
}

END_NAMESPACE_STIR

#endif // STIR_SIMPLE_BITMAPS

#ifdef STIR_MATHLINK

#include "mathlink.h"
extern "C" void init_and_connectlink( char* linkname);
extern "C" MLINK lp;

START_NAMESPACE_STIR

/* TODO, this ignores all arguments for the moment, except plane_stack and scale_factors */
template <class elemT, class SCALE, class CHARP>
void display_mathlink(const Array<3,elemT>& plane_stack,
	     const VectorWithOffset<SCALE>& scale_factors,
	     const VectorWithOffset<CHARP>& text,
	     double maxi, const char * const title, int scale)
{
  if (plane_stack.get_length() == 0)
    return;

  init_and_connectlink( "PARAPET");
  fprintf( stderr, "Writing data to MathLink\n");
  MLPutFunction( lp, "List", plane_stack.get_length());
  if( MLError( lp)) 
    fprintf( stderr, "Error detected by MathLink: %s.\n",
             MLErrorMessage(lp));
  int z = scale_factors.get_min_index();
  for (Array<3,elemT>::const_iterator iter1=plane_stack.begin(); 
       iter1!=plane_stack.end(); 
       iter1++, z++)
    {
      MLPutFunction( lp, "List", iter1->get_length());
      for (Array<2,elemT>::const_iterator iter2=iter1->begin(); iter2!=iter1->end(); iter2++)
	{
	  double *tmp = new double[iter2->get_length()];
	  int i = 0;
	  for (Array<1,elemT>::const_iterator iter3=iter2->begin(); iter3!=iter2->end(); iter3++)
	    tmp[i++] = static_cast<double>(*iter3)*scale_factors[z];
	  MLPutRealList(lp,tmp, static_cast<long>(iter2->get_length()));
	  delete [] tmp;	    
	}
    }
		
  if( MLError( lp)) 
	  fprintf( stderr, "Error detected by MathLink: %s.\n",
			MLErrorMessage(lp));
  MLEndPacket( lp);
  MLFlush( lp);
  if( MLError( lp)) 
	  fprintf( stderr, "Error detected by MathLink: %s.\n",
			MLErrorMessage(lp));

  /*MLPutFunction( lp, "Exit", 0);*/
}
 
END_NAMESPACE_STIR

#endif // STIR_MATHLINK

#ifdef STIR_PGM

#include <cstdio>

START_NAMESPACE_STIR


/* TODO, this ignores all arguments for the moment, except plane_stack and scale_factors */
template <class elemT, class SCALE, class CHARP>
void 
display_pgm (const Array<3,elemT>& plane_stack,
	     const VectorWithOffset<SCALE>& scale_factors,
	     const VectorWithOffset<CHARP>& text,
	     double maxi, const char * const title, int scale)
{
  if (plane_stack.get_length() == 0)
    return;
  
  Coordinate3D<int> min_indices;
  Coordinate3D<int> max_indices;

  if (!plane_stack.get_regular_range(min_indices, max_indices))
  {
    warning("display_pgm: can only display 'regular' arrays. Returning.\n");
    return;
  }

  char name[max_filename_length];
  ask_filename_with_extension(name, "Name for PGM file", ".pgm");
  
  FILE *pgm = fopen ( name , "wb");
  if (pgm == NULL)
  {
    warning("Error opening file %s for output to PGM.",name);
    return;
  }
  
  {
    int X = max_indices[3] - min_indices[3] + 1;    
    // for Y take into account we add 1 white line below every image
    int Y = (max_indices[2] - min_indices[2] + 2)*plane_stack.get_length();    
    fprintf ( pgm, "P5\n#created by PARAPET display \n%d %d\n255\n", X , Y);
  }
  
  double scaled_max;
  {    
    int z = min_indices[1];
    scaled_max = 
      static_cast<double>(plane_stack[z].find_max() * scale_factors[z]);
    for ( z++; z <= max_indices[1]; z++)
    { 
	  const double scaled_plane_max = 
		  static_cast<double>(plane_stack[z].find_max() * scale_factors[z]);
      if (scaled_max < scaled_plane_max)
		  scaled_max = scaled_plane_max;
    }
  }
  
  std::cerr << "Scaled maximum in image = " << scaled_max << std::endl;

  for ( int z = min_indices[1]; z <= max_indices[1]; z++)
  {
    for ( int y = min_indices[2]; y <= max_indices[2]; y++)
    {
      for ( int x = min_indices[3]; x <= max_indices[3]; x++)
      {
	double val = plane_stack[z][y][x]* scale_factors[z]*254. /scaled_max;
	int u = static_cast<int>(val+.5);
	fprintf ( pgm, "%c", u<0 ? 0 : u );
      }			  
    }
    // now draw white line below this image
    for ( int x = min_indices[3]; x <= max_indices[3]; x++)
    {
      fprintf ( pgm, "%c", 255 );
    }			  
  }
  fclose ( pgm);
  std::cerr<< "Wrote PGM plane_stack to file " <<  name << std::endl;
}

END_NAMESPACE_STIR

#endif // STIR_PGM


START_NAMESPACE_STIR

template <class elemT, class SCALE, class CHARP>
void display(const Array<3,elemT>& plane_stack,
	     const VectorWithOffset<SCALE>& scale_factors,
	     const VectorWithOffset<CHARP>& text,
	     double maxi, 
	     const char * const title, 
	     int scale)
{
  if (plane_stack.get_length() == 0)
    return;

  assert(plane_stack.get_min_index() == scale_factors.get_min_index());
  assert(plane_stack.get_max_index() == scale_factors.get_max_index());
  assert(plane_stack.get_min_index() == text.get_min_index());
  assert(plane_stack.get_max_index() == text.get_max_index());

  std::cerr << "Displaying " << (title==0 ? "" : title) << std::endl;
#if defined(STIR_PGM)
  display_pgm(plane_stack, scale_factors, 
                   text, maxi, title,  scale);
#endif

#if defined(STIR_SIMPLE_BITMAPS) && defined(STIR_MATHLINK)
  if (ask_num("Display as bitmap (0) or via MathLink (1)",0,1,0) == 0)
    display_bitmap(plane_stack, scale_factors, 
                   text, maxi, title, scale);
  else
    display_mathlink(plane_stack, scale_factors, 
                     text, maxi, title, scale);
#endif
#if defined(STIR_SIMPLE_BITMAPS) && !defined(STIR_MATHLINK)
  display_bitmap(plane_stack, scale_factors, 
                   text, maxi, title, scale);
#endif
#if !defined(STIR_SIMPLE_BITMAPS) && defined(STIR_MATHLINK)
  display_mathlink(plane_stack, scale_factors, 
                   text, maxi, title, scale);
#endif
}

template <class elemT>
void display(const RelatedViewgrams<elemT>& vs,
             double maxi,
	     const char * const title,
             int zoom)
{        
  Array<3,float> 
    all_of_them(IndexRange3D(0,vs.get_num_viewgrams()-1,
                             vs.get_min_axial_pos_num(),vs.get_max_axial_pos_num(), 
      	                     vs.get_min_tangential_pos_num(),vs.get_max_tangential_pos_num()));
    std::copy(vs.begin(), vs.end(), all_of_them.begin());
    
  VectorWithOffset<char *> text(all_of_them.get_min_index(), 
				all_of_them.get_max_index());
  
  VectorWithOffset<char *>::iterator text_iter = text.begin();
  typename RelatedViewgrams<elemT>::const_iterator vs_iter = vs.begin();

  while(vs_iter != vs.end())
  {
    *text_iter=new char[100];
    sprintf(*text_iter,"v %d, s %d", vs_iter->get_view_num(), vs_iter->get_segment_num());
    ++text_iter;
    ++vs_iter;
  }

  VectorWithOffset<float> scale_factors(all_of_them.get_min_index(), 
				        all_of_them.get_max_index());
  scale_factors.fill(1.);
  
  display(all_of_them, scale_factors, text,maxi, title, zoom);

  text_iter = text.begin();

  while(text_iter != text.end())
  {
    delete[] *text_iter;
    ++text_iter;
  }
}



/***********************************************
 instantiations
 ***********************************************/
template
void display<>(const Array<3,short>& plane_stack,
	     const VectorWithOffset<short>& scale_factors,
	     const VectorWithOffset<char *>& text,
	     double maxi, const char * const title, int scale);
template
void display<>(const Array<3,short>& plane_stack,
	     const VectorWithOffset<float>& scale_factors,
	     const VectorWithOffset<char *>& text,
	     double maxi, const char * const title, int scale);
template
void display<>(const Array<3,float>& plane_stack,
	     const VectorWithOffset<float>& scale_factors,
	     const VectorWithOffset<char *>& text,
	     double maxi, const char * const title, int scale);

template
void display<>(const Array<3,short>& plane_stack,
	     const VectorWithOffset<short>& scale_factors,
	     const VectorWithOffset<const char *>& text,
	     double maxi, const char * const title, int scale);
template
void display<>(const Array<3,short>& plane_stack,
	     const VectorWithOffset<float>& scale_factors,
	     const VectorWithOffset<const char *>& text,
	     double maxi, const char * const title, int scale);
template
void display<>(const Array<3,float>& plane_stack,
	     const VectorWithOffset<float>& scale_factors,
	     const VectorWithOffset<const char *>& text,
	     double maxi, const char * const title, int scale);


template
void display(const RelatedViewgrams<float>& vs,
             double maxi,
	     const char * const title,
             int zoom);

END_NAMESPACE_STIR
