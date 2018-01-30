/*!
 \file 
  
 \brief very basic display routines
  
 \author Kris Thielemans (with help from Claire Labbe)
 \author PARAPET project
 
  

 Implementations common to all SC_x types
  
 \see screen.h

 \internal
 
*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2001, IRSL
    See STIR/LICENSE.txt for details
*/
#include "gen.h"
#include <string.h>
#define SCreen_compiling
#include "screen.h"


void put_textstr (int  x, int y, const char str[])
{
  int  xx,yy;

  SC_MASK(SC_M_ANNOTATE);
  SC_COLOR(SC_C_ANNOTATE);
  SC_TJUST(2,1);
  xx = x;
  yy = y - 15;
  SC_MOVE(xx,yy);
  SC_TEXT(str);
  SC_TJUST(1,1);
  SC_MASK(SC_M_ALL);
}

/* TODO (?) scale should be obeyed if possible, now the smallest scale is used */
int center_sc_images(int *Pscale,
			   int min_x,int max_x,int min_y,int max_y,
                           int  SIZE_X,int SIZE_Y,
                           screen_image_t sc_image[], int nr_sc)
{ int   inc_x,inc_y,start_x,start_y;
  int   nr_x,nr_y,LENGTH_X,LENGTH_Y;
  int   nr,i,j,x,y,scale;

  /* how many times can we enlarge the images ?                         */
  LENGTH_X = max_x-min_x;               /* size in pixels of the region */
  LENGTH_Y = max_y-min_y;

  /* KT&CL 3/12/97 changed the test if 1 image fits in the window */
  if (LENGTH_X < SIZE_X || LENGTH_Y < SIZE_Y+15)
  { message("\nEven one image of this size doesn't fit in the window");
    return(0);
  }

  /* Now we find the maximum scale of the images */
  scale = 1;
  do
  {
    /* we compute the number of images of this scale that fit in the
       allowed region. Place for text under the images is reserved      */
    scale++;
    nr_x = LENGTH_X / (scale*(SIZE_X-1) + 1);
    nr_y = LENGTH_Y / (scale*(SIZE_Y-1) + 1+15);
  } while (nr_x*nr_y >= nr_sc);
  scale--;                              /* one too far                  */

  nr_x = LENGTH_X / (scale*(SIZE_X-1) + 1);
  nr_y = LENGTH_Y / (scale*(SIZE_Y-1) + 1+15);
  /* we want to center the images in the region
     eg: we have now computed that a maximum of 4x3 images fits
         but we have only 7 images ...                                  */
  /* KT 11/12/97 added first if to check if all images fit or not, and the else
     case to change nr_sc */
  if (nr_x*nr_y > nr_sc)
    {
      if (nr_y==1) nr_x=nr_sc;
      else
	{
	  nr_y = nr_sc / nr_x;
	  if (nr_x*nr_y < nr_sc) nr_y++;
	}
    }
  else
    {
      nr_sc = nr_x*nr_y;
    }

  if (*Pscale!=0 && scale>*Pscale)
      scale = *Pscale;
  *Pscale = scale;

  LENGTH_X = scale*(SIZE_X-1)+1;
  LENGTH_Y = scale*(SIZE_Y-1)+1;
  start_x = (min_x + max_x - nr_x*LENGTH_X)/2;
  start_y = (min_y + max_y - nr_y*(LENGTH_Y+15))/2;
  if (start_y<15)
    start_y = 15;
  inc_x   = (max_x-start_x) / nr_x;
  inc_y   = (max_y-start_y) / nr_y;
  start_y += inc_y*(nr_y-1);            /* begin on top of screen       */

  for(nr=0, i=0, y=start_y; i<nr_y; i++, y -=inc_y)
    for(j=0, x=start_x; j<nr_x && nr<nr_sc; j++, x +=inc_x, nr++)
    { sc_image[nr].sx = x;
      sc_image[nr].sy = y;
    }

  return(nr_sc);
}

void draw_sc_images(int size_x, int size_y,
                           screen_image_t sc_image[], int no)
{ int i;

  for (i=0; i<no; i++)
  {
    SC_PutImg(sc_image[i].image,sc_image[i].sx,sc_image[i].sy,size_x,size_y);
    put_textstr(sc_image[i].sx,sc_image[i].sy+15,sc_image[i].text); 
                                                                                                           
    
  }
  SC_FLUSH();
}
