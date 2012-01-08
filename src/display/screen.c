/* 
 $Id$
*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, IRSL
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
  
 \brief very basic display routines for bitmaps (internal use only)
  
 \author Kris Thielemans
 \author PARAPET project
 
 $Date$
  
 $Revision$

 \see screen.h for a few details
  
 \internal
 
*/

#include "gen.h"
#include <string.h>
#include <assert.h>
#include "screen.h"

#ifdef SC_XWINDOWS
	  
Display * SCX_display = NULL;
Window SCX_window;
GC SCX_gc;
XVisualInfo SCX_visual_info;
/* translation of linear color range SC_C_BACKGROUND...SC_C_FULL
   to X colors as stored in its color map.
   For Pseudocolor this is an identity mapping, but not for TrueColor
   (or any of the other visual classes presumably).
*/
unsigned long SCX_color_translation[SC_C_FULL+1];

static XPixmapFormatValues SCX_pixmap_format;

static Colormap SCX_Colormap;
static unsigned int CurrentColormap;

int SC__curPointX, SC__curPointY, SC__filled;
unsigned long SC__color;

/* Predefined Color scales */
#define NUMBER_SCALES 5

/* this gives RGB values, and the number of entries will be used for
   linear interpolation between this and the next entry.
   All .F values should add up to 128.
   Last entry is used for the annotation color.
*/
struct color_info { SC_pixel_t R,G,B,F;};

/* last entry is annotation color */
struct color_info sm4[6]={{0,0,0,32},{0,64,94,32},{0,127,0,32},
 {127,0,0,32},{127,127,127,0},{127,127,0,0}};

struct color_info sm4test[6]={{0,0,0,32},{0,0,127,32},{0,127,0,32},
 {127,0,0,32},{127,127,127,0},{127,127,0,0}};

/* change November 1997
   .F values added only up to 126, added 2 to the second entry
   */
struct color_info sm4green[6]={{0,0,0,32},{0,0,127,64},{0,127,0,31},
 {127,0,0,1},{127,127,127,0},{127,127,0,0}};

/* KT 28/11/2002 adjusted first .F value to 128 */
struct color_info bw[4]={{0,0,0,128},{127,127,127,0},{127,127,0,0}};

/* KT 29/01/98 added inverse greyscale */
struct color_info inverse_bw[4]={{127,127,127,128},{0,0,0,0},{127,127,0,0}};

/* KT 17/07/2000 added braces for inverse_bw */
struct
{ char name[10];
  int size;
  struct color_info *p;
} all_color_scales[NUMBER_SCALES]={{"sm4",5,sm4},{"sm4test",5,sm4test},
				   {"sm4green",5,sm4green},{"bw",2,bw},
				   {"inverse_bw",2,inverse_bw}};


/* KT 28/11/2002 heavily modified to account for TrueColor */
int CreateColormap(Display *mydisplay, XVisualInfo my_visual_info, struct color_info *x, int size)
{
  XColor cc;
  int   i,j,k;
  float tmp_r,tmp_g,tmp_b;
/* XWindows defines RGB values as unsigned 16-bit ints, the color scales above
   assume values between 0-127, we have to convert these
*/
  const float SCX_Color_conv = 60535. / 127;

  if (my_visual_info.visual == NULL)
    return 1;

  SCX_Colormap = XCreateColormap(mydisplay,DefaultRootWindow (SCX_display),
                                 my_visual_info.visual,
				 AllocNone
				 );
  cc.flags = DoRed | DoGreen |DoBlue;

  for (i=0,k=0;i<size-1;i++)
  {
    tmp_r=((int)x[i+1].R-(int)x[i].R)/(float)x[i].F;
    tmp_g=((int)x[i+1].G-(int)x[i].G)/(float)x[i].F;
    tmp_b=((int)x[i+1].B-(int)x[i].B)/(float)x[i].F;
    for (j=0;j<x[i].F;j++)
    {
      int old_pixel_value;
      cc.red     = (int)((x[i].R+tmp_r*j) * SCX_Color_conv);
      cc.green   = (int)((x[i].G+tmp_g*j) * SCX_Color_conv);
      cc.blue    = (int)((x[i].B+tmp_b*j) * SCX_Color_conv);
      cc.pixel = (int)(k++ * (SC_C_MAX-SC_C_BACKGROUND+1)/128. + .5)
                 + SC_C_BACKGROUND;
      assert(cc.pixel<=SC_C_MAX);
      old_pixel_value = cc.pixel;

      /*message("at %d asked %d,%d,%d, ", cc.pixel, cc.red, cc.green, cc.blue);*/
  
      if (XAllocColor (mydisplay, SCX_Colormap, &cc) == 0)
      {
        message("X colormap initialisation: Error in color alloc %d, value %lu\n",
		 k, cc.pixel);
	/*    return (1);*/
      }
      /* necessary for Truecolor */
      SCX_color_translation[old_pixel_value] = cc.pixel;
      /*message("got %d,%d,%d pix %lx\n", cc.red, cc.green, cc.blue, cc.pixel);*/
    }
  }


  /* fill in last half of colors with the Annotation color
     for masking */
  cc.red     = (int)((x[size].R) * SCX_Color_conv);
  cc.green   = (int)((x[size].G) * SCX_Color_conv);
  cc.blue    = (int)((x[size].B) * SCX_Color_conv);
  for (k=SC_C_MAX+1; k<=SC_C_FULL; k++)
  {
    cc.pixel = k;
    if (XAllocColor (mydisplay, SCX_Colormap, &cc) == 0)
    {
        message("X colormap initialisation: Error in color alloc %d, value %lu\n",
		 k, cc.pixel);
  /*    return (1);*/
    }
    SCX_color_translation[k] = cc.pixel;
  }
  return 0;
}

int SetColormap(Display *mydisplay, Window  mywindow, XVisualInfo my_visual_info, struct color_info *x, int size)
{
  CreateColormap(mydisplay, my_visual_info, x, size);
  XSetForeground (mydisplay, SCX_gc, SC_C_ANNOTATE);
  XSetWindowColormap (mydisplay, mywindow, SCX_Colormap);
  XSetWindowBorder (mydisplay, mywindow, SC_C_ANNOTATE);
  return (0);

}

/* KT 28/11/2002 heavily modified to account for TrueColor */
/* helper function called by SC?_START() */
static void SCX_init(XSizeHints* SCX_hint)
{
  SCX_display = XOpenDisplay ("");
  /* KT 13/10/98 added some error checking*/
  if (!SCX_display) 
    {
        error("Cannot open DISPLAY\n");
    }
  /* KT 25/01/2000 extra error check */
  if (ScreenCount(SCX_display)==0)
    {
        error("Cannot open DISPLAY: screen count = 0\n");
    }

  /* find appropriate visual */
  {
   /* I prefer to use PseudoColor whenever the X server supports it.
      Strategy:
      try PseudoColor, then TrueColor, then the default visual.
   */
    if (!XMatchVisualInfo(SCX_display, DefaultScreen(SCX_display),
	  8, PseudoColor, &SCX_visual_info) &&
        !XMatchVisualInfo(SCX_display, DefaultScreen(SCX_display),
			  24, TrueColor, &SCX_visual_info) &&
        !XMatchVisualInfo(SCX_display, DefaultScreen(SCX_display),
			  16, TrueColor, &SCX_visual_info) &&
        !XMatchVisualInfo(SCX_display, DefaultScreen(SCX_display),
			  15, TrueColor, &SCX_visual_info))
    {
      /* get default */
      XVisualInfo* vinfo_list;
      int nitems;
      SCX_visual_info.visualid = 
	XVisualIDFromVisual(DefaultVisual(SCX_display,DefaultScreen(SCX_display)));      
#if 0
      /* enable this when you want to be able to select the visual. If so,
         you can get the required number by using xdpyinfo.
      */
      SCX_visual_info.visualid = 
	asknr("VisualID",0,10000,SCX_visual_info.visualid);
#endif
      vinfo_list = 
	XGetVisualInfo(SCX_display, VisualIDMask, &SCX_visual_info, &nitems);
      if (nitems==0)
	{
	  error("X: Cannot get VisualInfo from default visual. Sorry\n");
	}
      SCX_visual_info = vinfo_list[0];
      XFree(vinfo_list);
    }


    /* write some info about the visual to the screen */
    {
      char *vclass;
      switch (SCX_get_class(SCX_visual_info))
	{
	case StaticGray: vclass = "StaticGray"; break;
	case GrayScale: vclass = "GrayScale"; break;
	case StaticColor: vclass = "StaticColor"; break;
	case PseudoColor: vclass = "PseudoColor"; break;
	case TrueColor: vclass = "TrueColor"; break;
	case DirectColor: vclass = "DirectColor"; break;
	default: vclass = "Unknown"; break;
	}
      message("X visual selected: depth %d with class %s", 
	      SCX_visual_info.depth, vclass);
#if 0
      message("red_mask %lx\ngreen_mask %lx\nblue mask %lx\n",
	      SCX_visual_info.red_mask, SCX_visual_info.green_mask, SCX_visual_info.blue_mask);
#endif
    }
  }

  /* find appropriate ZPixmap format */
  {
    int nitems, i;
    XPixmapFormatValues * pixmapformats =
      XListPixmapFormats(SCX_display, &nitems);
    if (pixmapformats == NULL)
      {
	message("X out of memory when requesting Pixmap info. No display of bitmaps\n");
	SCX_pixmap_format.depth = 0;
      }
    if (nitems==0)
      {
	message("X does not support any pixmap formats (?). No display of bitmaps\n");
	SCX_pixmap_format.depth = 0;
      }
    for (i=0; i<nitems; ++i)
      {
	if (pixmapformats[i].depth==SCX_visual_info.depth)  
	  {
	    SCX_pixmap_format = pixmapformats[i];
	    break;
	  }
      }
#if 1
    message("Found pixmap format with depth %d, bits_per_pixel %d, scanline_pad %d.",
	    SCX_pixmap_format.depth, SCX_pixmap_format.bits_per_pixel, 
	    SCX_pixmap_format.scanline_pad);
#endif
    XFree(pixmapformats);
  } /* end ZPixmap format */

  CurrentColormap = NUMBER_SCALES-2; /* default to 'bw' colormap */
  /* KT 28/11/2002 ask for color map if it's not a PseudoColor visual as 
     we cannot switch it without redrawing*/
  if (SCX_get_class(SCX_visual_info) != PseudoColor)
    {
      int i;
      message("Color scales:");
      for (i=0; i<NUMBER_SCALES; i++)
	printf("%d: %s|",i,all_color_scales[i].name);
      CurrentColormap = 
	asknr("Which one do you want ?",0,NUMBER_SCALES-1, CurrentColormap);
    }  
  message("Color map selected: %s\n", all_color_scales[CurrentColormap].name);

  CreateColormap(SCX_display, SCX_visual_info,
	      all_color_scales[CurrentColormap].p, 
	      all_color_scales[CurrentColormap].size);
  

  /* create window */
  {
    XSetWindowAttributes attrib;
    attrib.backing_store = Always;
    attrib.colormap = SCX_Colormap;
    /* Note: the next 2 are necessary, or you get a BadMatch error.
     Thanks to Ken Lee to helping me out on this one. */
    attrib.backing_pixel = SCX_color_translation[SC_C_BACKGROUND];
    attrib.border_pixel = SCX_color_translation[SC_C_ANNOTATE];
    SCX_window = XCreateWindow (
				SCX_display, DefaultRootWindow (SCX_display),
				SCX_hint->x, SCX_hint->y, 
				(unsigned)SCX_hint->width,  (unsigned)SCX_hint->height, 5, 
				SCX_visual_info.depth, InputOutput,
				SCX_visual_info.visual,
				CWBackingStore|CWColormap|CWBorderPixel|CWBackingPixel, 
				&attrib);

    XSetStandardProperties (SCX_display, SCX_window,
			    "Display", "Display", None, 0, 0, SCX_hint);
  }

  SCX_gc = XCreateGC (SCX_display, SCX_window, 0, 0);

  XMapWindow (SCX_display, SCX_window);
  XSync(SCX_display, False);

  /* first set mask to everything, in case we're not in PseudoColor */
  XSetPlaneMask(SCX_display, SCX_gc, (unsigned long)-1);
  SC_COLOR(SC_C_ANNOTATE);
  SC_MASK(SC_M_ANNOTATE);
}

#define SCX_hintX 800
#define SCX_hintY 600

void SCX_START()
{
  XSizeHints SCX_hint;

  SCX_hint.x = 50;
  SCX_hint.y = 50;
  SCX_hint.width = SCX_hintX;
  SCX_hint.height = SCX_hintY;
  SCX_hint.flags = PPosition | PSize;

  SCX_init(&SCX_hint);
}

void SCX_START_BIG()
{
  XSizeHints SCX_hint;
  SCX_hint.x = (unsigned int) SCREEN_X_MAX/100;
  SCX_hint.y = (unsigned int) SCREEN_Y_MAX/100;
  SCX_hint.width = (unsigned int) SCREEN_X_MAX*9/10;
  SCX_hint.height = (unsigned int) SCREEN_Y_MAX*9/10;
  SCX_hint.flags = PPosition | PSize;

  SCX_init(&SCX_hint);
}

/* Change April 1997: 
   changed two occurences of XLookupString(&report,...) to
   XLookupString(&report.xkey,...)
   It worked previously because report is a union including the correct type.
*/
int SCX_WRITE(int *Xpos_x,int *Xpos_y,char *text)
{ /* when deleting characters, pieces of the image could disappear ?? */
  SC_MASK(SC_M_ANNOTATE);
  XSelectInput (SCX_display, SCX_window,
         ButtonPressMask | KeyPressMask);
  while (1)
  { XEvent report;
    char chr[2];
    XNextEvent (SCX_display, &report);
    switch (report.type)
    { case ButtonPress:
        if (report.xbutton.button == Button3)
        { int x,y;
          int stop = 0;
          int nr_chr = 0;
          x = *Xpos_x = report.xbutton.x ;
          y = *Xpos_y = report.xbutton.y ;
          while (!stop)
          { XNextEvent (SCX_display, &report);
            if (report.type == KeyPress)
            {
              XLookupString(&report.xkey, &chr[0], 1, NULL, NULL);
              chr[1] = '\0';
              switch (chr[0])
              { case  13: /* CR */
                  text[nr_chr+1] = '\0';
                  stop = 1;
                  break;
                case '\177': /* BS ???? change it */
                  if (nr_chr )
                  { chr[0] = text[--nr_chr];
                    x -= SC_CHAR_SPACING;
                    SC_MOVE(x,y);
                    SC_COLOR(SC_C_BACKGROUND);
                    SC_TEXT(chr);
                    SC_COLOR(SC_C_ANNOTATE);
                  }
                  break;
                case 27: /* ESC */
                  while(nr_chr)
                  { chr[0] = text[--nr_chr];
                    x -= SC_CHAR_SPACING;
                    SC_MOVE(x,y);
                    SC_COLOR(SC_C_BACKGROUND);
                    SC_TEXT(chr);
                    SC_COLOR(SC_C_ANNOTATE);
                  }
                  break;
                default :
                  SC_MOVE(x,y);
                  SC_TEXT(chr);
                  x += SC_CHAR_SPACING;
                  text[nr_chr++] = chr[0];
                  break;
              }
            }
          }
        }
        break;
      case KeyPress:
        XLookupString(&report.xkey, &chr[0], 1, NULL, NULL);
        if (chr[0] == 'q' || chr[0] == 'Q')
        { SC_MASK(SC_M_ANNOTATE);
          return (0);
        }
        else return(1);
        break;
    }
  }
}

void SCX_STOP(stop)
int stop;
{ XEvent report;

/* KT 28/11/2002 do this only for PseudoColor visuals. For the others, it is a noop */
 if (SCX_get_class(SCX_visual_info)==PseudoColor)
   {
     message("To change the colormap, press mouse button 2 and\n"
	     "to stop the display, press a key.\n"
	     "Both only work while the display window is selected.\n");
     XSelectInput (SCX_display, SCX_window, KeyPressMask | ButtonPressMask);
   }
 else
     XSelectInput (SCX_display, SCX_window, KeyPressMask);

  while (!stop)
  {
    XNextEvent (SCX_display, &report);
    switch (report.type)
    {
      case KeyPress:
        stop = 1;
        break;
      case ButtonPress:
	if (report.xbutton.button != Button2) break;
	CurrentColormap = (CurrentColormap+1) % NUMBER_SCALES;

	SetColormap(SCX_display, SCX_window, SCX_visual_info,
		    all_color_scales[CurrentColormap].p, 
		    all_color_scales[CurrentColormap].size);
	break;
      
    }
  }
  XUnmapWindow (SCX_display, SCX_window);
  XFreeGC (SCX_display, SCX_gc);
  XFreeColormap (SCX_display, SCX_Colormap);
  XDestroyWindow (SCX_display, SCX_window);
  XCloseDisplay (SCX_display);
  SCX_display = NULL;
}

unsigned SCX_X_MAX()
{ Window wdummy;
/*KT 13/10/98 use different variables to prevent conflicts*/
  int x,y;
  unsigned width, height, bw, dep;

  if (SCX_display)   
    XGetGeometry(SCX_display, SCX_window, &wdummy,
                 &x, &y, &width, &height, &bw, &dep);
  else
    width = SCX_hintX;
  return (width);
}

unsigned SCX_Y_MAX()
{ Window wdummy;
/*KT 13/10/98 use different variables to prevent conflicts*/
  int x,y;
  unsigned width, height, bw, dep;

  if (SCX_display)
    XGetGeometry(SCX_display, SCX_window, &wdummy,
            	 &x, &y, &width, &height, &bw, &dep);
  else
    height = SCX_hintY;
  return (height);
}

#undef SCX_hintX
#undef SCX_hintY

/* KT 28/11/2002 heavily modified to account for TrueColor */
void SCX_PutImg (image,x_begin,y_begin,lengthX,lengthY)
image_t *image;
int x_begin,y_begin, lengthX,lengthY;
{
  XImage * myimage;
  unsigned char * local_image;
  int bytes_per_line;

  if (SCX_pixmap_format.depth == 0)
    return;

  bytes_per_line = 
      ((lengthX*SCX_pixmap_format.bits_per_pixel + 
	SCX_pixmap_format.scanline_pad-1
	)/SCX_pixmap_format.scanline_pad
       ) * (SCX_pixmap_format.scanline_pad/8);

  if (SCX_get_class(SCX_visual_info)!=PseudoColor || bytes_per_line != lengthX )
    {
      /* we'll need to copy the image somewhere else, so reserve space for it*/
      local_image =malloc(bytes_per_line*lengthY);
      if (local_image==NULL)
	{
	  message("SCX_PutImg: cannot allocate enough memory. I cannot display this bitmap\n");
	  return;
	}
    }
  else
    { 
      local_image = (unsigned char*)image;
    }
      

  if (SCX_get_class(SCX_visual_info)!=PseudoColor || bytes_per_line != lengthX )
  {
    int i;

    for (i=0; i<  lengthX*lengthY; ++i)
    {
      unsigned long color;
      unsigned char * current_position =
	local_image + (i/lengthX)*bytes_per_line + 
	(i%lengthX)*(SCX_pixmap_format.bits_per_pixel/8);
      assert(SCX_pixmap_format.bits_per_pixel%8 == 0 );
      assert(image[i]>=SC_C_BACKGROUND);
      assert(image[i]<=SC_C_MAX);
      color = SCX_color_translation[image[i]];

      /* Now store the color in the pixmap, taking care of bits_per_pixel and byte_order.
	 Code adapted by KT from a very helpful example by Kip Rugger.
      */
      if (SCX_pixmap_format.bits_per_pixel == 8) 
        ((unsigned char*)current_position)[0] = (unsigned char)color;
      else if (SCX_pixmap_format.bits_per_pixel == 16)
	{
	  if (ImageByteOrder(SCX_display)!=MSBFirst)
	    {
	      current_position[0] = color; 
	      current_position[1] = color>>8; 
	    }
	  else
	    {
	      current_position[1] = color; 
	      current_position[0] = color>>8; 
	    }
	  assert(color>>16==0);	
	}
      else if (SCX_pixmap_format.bits_per_pixel == 24)
	{ 
	  if (ImageByteOrder(SCX_display)!=MSBFirst)
	    {
	      current_position[0] = color; 
	      current_position[1] = color>>8; 
	      current_position[2] = color>>16; 
	    }
	  else
	    {
	      current_position[2] = color; 
	      current_position[1] = color>>8; 
	      current_position[0] = color>>16; 
	    }
	  assert(color>>24 == 0);
	}
      else if (SCX_pixmap_format.bits_per_pixel == 32)
	{
	  if (ImageByteOrder(SCX_display)!=MSBFirst)
	    {
	      current_position[0] = color; 
	      current_position[1] = color>>8; 
	      current_position[2] = color>>16; 
	      current_position[3] = color>>24; 
	    }
	  else
	    {
	      current_position[3] = color; 
	      current_position[2] = color>>8; 
	      current_position[1] = color>>16; 
	      current_position[0] = color>>24; 
	    }
	}
      else 
	{
	  message("SCX_PutImg: bits_per_pixel (%d) does not match 8,16,24 nor 32\n"
		  "I cannot display the bitmap. Sorry\n",
		  SCX_pixmap_format.bits_per_pixel);
	  return;
	}
    }
    free(image);
  
  } /* !PseudoColor */

  myimage = XCreateImage (SCX_display,
			  SCX_visual_info.visual,
			  SCX_pixmap_format.depth,
			  ZPixmap, /*offset */ 0, (char *)local_image, 
			  (unsigned)lengthX, (unsigned)lengthY, 
			  SCX_pixmap_format.scanline_pad, bytes_per_line);
  if (myimage == NULL)
    {
      message("XcreateImage returned 0. No bitmap displayed.\n");
      return;
    }
  assert(myimage->byte_order == ImageByteOrder(SCX_display));
#if 0
  message("XImage props.:\nwidth %d\nheight %d\\n xoffset %d\nbitmap_unit %d\nbitmap_pad %d\ndepth %d\n"
          "bytes_per_line %d\nbits_per_pixel %d\n"
	  " byte_order: %s\n",
          myimage->width,myimage->height, myimage->xoffset,
          myimage->bitmap_unit, myimage->bitmap_pad, myimage->depth,
          myimage->bytes_per_line, myimage->bits_per_pixel,
	  myimage->byte_order==MSBFirst ? "MSBFirst" : "LSBFirst");
#endif
       
  SC_MASK(SC_M_ALL);
  XPutImage (SCX_display, SCX_window, SCX_gc, myimage,
         0, 0, x_begin, y_begin, (unsigned)lengthX, (unsigned)lengthY);
  SC_MASK(SC_C_ANNOTATE);
  XDestroyImage(myimage);  /* Note: this deallocates local_image as well */
}

void SCX_SAVE_TO_FILE
(int x_begin,int y_begin,int width,int height,FILE *outfile)
{ XImage * myimage;
  char  *proc = "SCX_SAVE_TO_FILE";
/*
  char  *dat;

  myimage = XCreateImage (SCX_display,
      XDefaultVisual (SCX_display, DefaultScreen (SCX_display)),
      DefaultDepthOfScreen(DefaultScreenOfDisplay(SCX_display)),
      ZPixmap, 0, dat, (int) width, (int)height, 8, 0);
*/

  myimage = XGetImage(SCX_display, SCX_window,x_begin,y_begin,
                width,height, AllPlanes,ZPixmap);
  fwrite_check(proc,(void *)myimage->data,
        (unsigned long)sizeof(SC_pixel_t)*width*height,outfile);
}

#endif /* SC_XWINDOWS */

/* change November 1997: added this function (was SCX_SCALE before) */
/* 30/01/98 put high intensities on top of scale 
   25/11/2002 try this again*/
void SC_SCALE(pos_x,pos_y,size_x,size_y)
int pos_x, pos_y, size_x, size_y;
{
  unsigned char par;
  float pos_inc;  
  float pos_offset;

  assert(pos_y+size_y<=SC_Y_MAX);
  assert(pos_x+size_x<=SC_X_MAX);
  SC_MASK(SC_M_ALL);
  SC_PRMFIL(1);
  SC_MOVE(pos_x,pos_y);

  /* KT 17/06/2000 condition now uses && instead of a ,*/
  /* find pos_offset and pos_inc such that y = pos_offset+par*pos_inc
     varies between pos_y+size_y and pos_y for par between
     SC_C_BACKGROUND and SC_C_MAX */
  pos_inc = -((float)size_y) / (SC_C_MAX-SC_C_BACKGROUND+1);
  pos_offset = pos_y+size_y - pos_inc*SC_C_BACKGROUND;
  for(par=SC_C_BACKGROUND;
      par<=SC_C_MAX;
      par++)
    { SC_COLOR(par);
      SC_RECT(pos_x+size_x,(int)(pos_offset+pos_inc*par+.5));
    }
  SC_PRMFIL(0);
  SC_MASK(SC_C_ANNOTATE);
  /* change November 1997: added rectangle around the color scale */
  SC_COLOR(SC_C_ANNOTATE);
  SC_MOVE(pos_x, pos_y);
  SC_RECT(pos_x+size_x, pos_y+size_y);
}
