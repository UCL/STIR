/*!
 \file
  
 \brief very basic display routines for bitmaps (internal use only)
  
 \author Kris Thielemans
 \author PARAPET project
 
 $Date$
 $Revision$

 This is part of a library by Kris Thielemans, mainly written in 1991.
 It provides macros (and a few functions) for displaying stuff on a screen.
 The files only contain the macros for XWindows.
 It's fairly simplistic. No menus. Just simple display of bitmaps,
 lines, points, text strings.
  
 \internal
 
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


/* Change November 1997: added next 3 lines */
#ifdef __cplusplus
extern "C" {
#endif

#ifdef __STDC__
#define ANSI
#endif


#ifndef SC_pixel_t
#define SC_pixel_t unsigned char

/* Change 13/02/98 */
#ifndef MSDOS16BIT
typedef SC_pixel_t image_t;
#else
typedef SC_pixel_t huge image_t;
#endif
#endif /* SC_pixel_t */


#ifdef SC_XWINDOWS
#include <X11/Xlib.h>
#include <X11/keysym.h>
#include <X11/Xutil.h>

#define SC_C_BACKGROUND 12
#define SC_C_MAX	127		/* maximum color to be used */
#define SC_C_FULL	255		/* maximum color available */

extern Display * SCX_display;
extern Window SCX_window;
extern GC SCX_gc;
  /* KT 28/11/2002 added for TrueColor support */
extern XVisualInfo SCX_visual_info;
extern unsigned long SCX_color_translation[SC_C_FULL+1];

extern int SC__curPointX, SC__curPointY, SC__filled;
extern unsigned long SC__color;
extern unsigned SCX_X_MAX(void), SCX_Y_MAX(void);

/* define macro to access XVisualInfo.class. 
   This is renamed by X to c_class when compiling C++ programs.
*/
#if defined(__cplusplus) || defined(c_plusplus)
#define SCX_get_class(vinfo)	  vinfo.c_class
#else
#define SCX_get_class(vinfo)	  vinfo.class
#endif

extern void SCX_START(void);
/* KT 01/03/2000 added next declaration */
extern void SCX_START_BIG(void);
extern void SCX_STOP(int stop);
extern int SCX_WRITE(int *x,int *y,char *text);
extern void SCX_PutImg     (image_t *, int x_begin, int y_begin,
                                int lengthX, int lengthY);

#define SCREEN_X_MAX    1024	/* VR299 max ??*/
#define SCREEN_Y_MAX    864	/* VR299 max */
#define SC_START()      SCX_START()
#define SC_START_BIG()  SCX_START_BIG()
#define SC_STOP()       SCX_STOP(0)
#define SC_STOP_CLEAR() SCX_STOP(1)
#define SC_FLUSH()      XSync(SCX_display, False)
#define SC_X_MAX        SCX_X_MAX()
#define SC_Y_MAX        SCX_Y_MAX()
#define SC_CHAR_WIDTH   10
#define SC_CHAR_HEIGHT  10
#define SC_CHAR_SPACING 7
#define SC_PutImg(image,x,y,lx,ly) \
                        SCX_PutImg(image,x,y,lx,ly)
#define SC_TJUST(hor,ver)
#define SC_TSTYLE(par)
#define SC_PRMFIL(par)  SC__filled = par
#define SC_COLOR(par)   XSetForeground(SCX_display, SCX_gc,\
                                SC__color = (unsigned long)SCX_color_translation[par])
#define SC_TSIZE(par)
#define SC_POINT()      XDrawPoint(SCX_display,SCX_window,SCX_gc,\
                                SC__curPointX, SC__curPointY)
#define SC_MOVE(x,y)    { SC__curPointX = (int)(x); SC__curPointY = (int)(y);}
#define SC_DRAW(x,y)    { XDrawLine(SCX_display,SCX_window,SCX_gc,\
                                SC__curPointX, SC__curPointY, (int)(x),(int)(y));\
                          SC_MOVE(x,y);\
                        }
#define SC_LINE(x1,y1,x2,y2)\
			{ XDrawLine(SCX_display,SCX_window,SCX_gc,\
                                (int)(x1), (int)(y1),\
				SC__curPointX=(int)(x2),\
				SC__curPointY=(int)(y2));\
                        }
#define SC_RECT(x,y)    ( SC__filled ? \
                          XFillRectangle(SCX_display, SCX_window, SCX_gc,\
                                (int)Min(x,SC__curPointX), (int)Min(y,SC__curPointY),\
                                (unsigned int)abs(x-SC__curPointX), (unsigned int)abs(y-SC__curPointY))\
                        : XDrawRectangle(SCX_display, SCX_window, SCX_gc,\
                                (int)Min(x,SC__curPointX), (int)Min(y,SC__curPointY),\
                                (unsigned int)abs(x-SC__curPointX), (unsigned int)abs(y-SC__curPointY))\
                        )
#define SC_RECTR(x,y)   ( SC__filled ? \
                          XFillRectangle(SCX_display, SCX_window, SCX_gc,\
                                (x<0 ? SC__curPointX + x : SC__curPointX),\
                                (y<0 ? SC__curPointY + y : SC__curPointY),\
                                abs(x), abs(y))\
                        : XDrawRectangle(SCX_display, SCX_window, SCX_gc,\
                                (x<0 ? SC__curPointX + x : SC__curPointX),\
                                (y<0 ? SC__curPointY + y : SC__curPointY),\
                                abs(x), abs(y))\
                        )
#define SC_ELLIPSE(x,y)  ( SC__filled ? \
                          XFillArc(SCX_display, SCX_window,SCX_gc, \
                                SC__curPointX-x,SC__curPointY-y,\
                                2*x, 2*y, 0, 360*64)\
                        : XDrawArc(SCX_display, SCX_window,SCX_gc, \
                                SC__curPointX-x,SC__curPointY-y,\
                                2*x, 2*y, 0, 360*64)\
                        )
#define SC_CIRCLE(x)    SC_ELLIPSE(x,x)
#define SC_TEXT(str)    XDrawString(SCX_display, SCX_window,SCX_gc, \
                                SC__curPointX, SC__curPointY, \
                                str, (int)strlen(str))
  /* KT 28/11/2002 only enable mask when the visual is PseudoColor */
#define SC_MASK(par)    if (SCX_get_class(SCX_visual_info)==PseudoColor) \
                            XSetPlaneMask(SCX_display, SCX_gc, \
                                (unsigned long)par)
#define SC_LINFUN(par)  XSetFunction(SCX_display, SCX_gc, par)
#define SC_LUTX(par,R,G,B)
#define SC_CLEARS(par)
#define SC_LUTINT(par)
#define SC_CLEAR_BLOCK(color,x_b,x_e,y_b,y_e)  \
			XSetForeground(SCX_display, SCX_gc,\
                                (unsigned long)SCX_color_translation[color]); \
  			XFillRectangle(SCX_display, SCX_window, SCX_gc, \
                                Min((x_b),(x_e)), Min((y_b),(y_e)),\
                                abs((x_b)-(x_e)), abs((y_b)-(y_e))); \
			XSetForeground(SCX_display, SCX_gc,\
                                (unsigned long)SC__color);
#define SC_LF_REPLACE   GXcopy
#define SC_LF_XOR       GXxor
#define SC_DEPTH	SCX_visual_info.depth
#define SC_C_ANNOTATE   (SC_C_MAX+1)
/* two definitions for masking availability */
/* Note: On X-windows, masking works only properly for a PseudoColor visual
   (i.e. 8-bit color with adjustable colormap)
*/  
  /* KT 28/11/2002 changed value for TrueColor support */
#define SC_M_ALL        ((unsigned long)-1L)
#define SC_M_ANNOTATE   (SC_C_MAX+1)

#endif /* SC_XWINDOWS */


/* change November 1997: added this function (was SCX_SCALE before) */
extern void SC_SCALE(int pos_x, int pos_y, int size_x,int size_y);

typedef struct screen_image
        { image_t *image;
          int sx,sy;
          char *text;
        } screen_image_t;
/*****************************************************************************
	 routines found in screengen.c
*****************************************************************************/
/* KT 01/03/2000 added const */
extern void put_textstr   (int x, int y, const char * str);
extern int  center_sc_images(int *Pscale,
			   int min_x,int max_x,int min_y,int max_y,
                           int  SIZE_X,int SIZE_Y,
                           screen_image_t sc_image[], int nr_sc);
extern void draw_sc_images(int size_x, int size_y,
                           screen_image_t sc_image[], int no);


/* Change November 1997: added next 3 lines: end of extern "C" */
#ifdef __cplusplus
}
#endif
