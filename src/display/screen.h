/*!
 \file
  
 \brief very basic display routines for bitmaps (internal use only)
  
 \author Kris Thielemans
 \author PARAPET project
 
 $Date$
 $Revision$

 This is part of a library by Kris Thielemans, mainly written in 1991.
 It provides macros (and a few functions) for displaying stuff on a screen.
 It works in XWindows, DOS (upto SVGA resolution, or using a PGA) and 
 a VAX using a Matrox card (very much similar to PGA).
 It's fairly simplistic. No menus. Just simple display of bitmaps,
 lines, points, text strings.
  
 \internal
 
*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, IRSL
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

#ifdef SC_QG

#include "stir/scqg.h"
typedef SC_pixel_t out_t;
#ifdef VAX
#define PGA "qga0:"
#else
#define PGA "cgb"
#endif

#ifndef SCreen_compiling                /*somewhat dirty trick to prevent*/
out_t buffer[OUTSIZE],*outptr;          /*having these also in screen*.obj*/
#else                                   /*only defined in screen*.c      */
extern out_t buffer[], *outptr;
#endif

#define SC_X_MAX 640
#define SC_Y_MAX 480
#define HEX   1
#define ASCII 0
#define RLCMAX 128
        /* Max length of a Run Length Encoded sequence */

extern SC_INIT(int);
        /* Initialization of the graphical screen.
           Sets whole screen as viewport
           lower left corner (0,0), upper right (SC_X_MAX,SC_Y_MAX)
           All bit planes writeable
           color: SC_C_ANNOTATE (for text purposes)

           Note: int parameter is HEX or ASCII */
extern SC_INIT_FILE(int,char []);
        /* As above, but takes care that subsequent commands for the screen
           will be written to file (name as seconde parameter) also */
extern SC_CLOSE(int);
extern SC_FLUSH(void);
        /* After this function all previous SC commands are executed */
extern void SCQ_PutImg     (image_t *, int x_begin, int y_begin,
                                int lengthX, int lengthY);
/* Set next variable to 1 if you want run length encoding.
   You get only better performance when rendering images with large
   blocks of one color
*/
extern int  SCQ_rl;

#ifdef VAX
extern void error_check(int sys_function);
extern void write_buffer(char *dev,char *buffer);
extern SC_pixel_t *read_buffer(char *dev,int size);
#endif

#define OUTB(b)         *outptr++= (SC_pixel_t)b
#define OUTW(w)         *outptr++=(SC_pixel_t)(((short)w)%256); \
                        *outptr++=(SC_pixel_t)(((short)w)/256)
#define OUTDW(w1,w2)    OUTW(w1); OUTW(w2)
#define OUTS(str)       strncpy(outptr,str,strlen(str));    outptr += strlen(str)
/* KT 12/01/2000 forget about fmemcpy (old compilers) */
#define OUTM(buf,nr)    memcpy(outptr,buf,nr); outptr += nr


#define SC_START() SC_INIT(HEX)
#define SC_START_BIG() SC_INIT(HEX)
#define SC_START_FILE(str) SC_INIT_FILE(HEX,str)
#define SC_STOP()  SC_CLOSE(ASCII)
#define SC_STOP_CLEAR() SC_STOP()
#define SC_LF_REPLACE   0
#define SC_LF_XOR       2
#define SC_DEPTH	8		/* 256 color screen */
#define SC_C_ANNOTATE   0x80
#define SC_C_BACKGROUND 0
#define SC_C_MAX	127		/* maximum color to be used */
#define SC_C_FULL	255		/* maximum color available */
/* two definitions for masking availability */
#define SC_M_ALL        0xff
#define SC_M_ANNOTATE   0x80

/* QG-640 (or PGA) have (0,0) point in lower left point of screen
   The following macro puts it here needed in the top left corner
*/
#define SC__Y(y)                SC_Y_MAX-1 - (y)

#define SC_PutImg(image,x,y,lx,ly) \
                        SCQ_PutImg(image,x,SC__Y(y),lx,ly)
#define SC_TJUST(hor,ver)       OUTB(TJUST);  OUTB(hor);OUTB(ver);
#define SC_TSTYLE(par)          OUTB(TSTYLE); OUTB(par)
#define SC_PRMFIL(par)          OUTB(PRMFIL); OUTB(par)
#define SC_COLOR(par)           OUTB(COLOR);  OUTB(par)
#define SC_TSIZE(par)           OUTB(TSIZE);  OUTDW(par,0)
#define SC_POINT()              OUTB(POINT);
#define SC_CIRCLE(x)            OUTB(CIRCLE); OUTDW(x,0);
#define SC_MOVE(x,y)            OUTB(MOVE);   OUTDW(x,0); OUTDW(SC__Y(y),0)
#define SC_DRAW(x,y)            OUTB(DRAW);   OUTDW(x,0); OUTDW(SC__Y(y),0)
#define SC_LINE(x1,y1,x2,y2)    {SC_MOVE(x1,y1); SC_DRAW(x2,y2); }
#define SC_RECT(x,y)            OUTB(RECT);   OUTDW(x,0); OUTDW(SC__Y(y),0)
#define SC_RECTR(x,y)           OUTB(RECTR);   OUTDW(x,0); OUTDW(SC__Y(y),0)
#define SC_ELLIPSE(x,y)         OUTB(ELIPSE);   OUTDW(x,0); OUTDW(SC__Y(y),0)
#define SC_TEXT(str)            OUTB(TEXT);   OUTB('"'); OUTS(str); OUTB('"')
#define SC_MASK(par)            OUTB(MASK);   OUTB(par)
#define SC_LINFUN(par)          OUTB(LINFUN);   OUTB(par)
#define SC_FLOOD(par)           OUTB(FLOOD);  OUTB(par)
#define SC_LUTX(par,R,G,B)      OUTB(LUTX);   OUTB(par);OUTB(R);OUTB(G);OUTB(B)
#define SC_IMAGEW(line,begin,end) OUTB(IMAGEW); OUTW(line);OUTW(begin);OUTW(end)
#define SC_IMAGER(line,begin,end) OUTB(IMAGER); OUTW(line);OUTW(begin);OUTW(end)
#define SC_RESETF               OUTB(RESETF)
#define SC_CLEARS(par)          OUTB(CLEARS); OUTB(par)
#define SC_LUTINT(par)          OUTB(LUTINT); OUTB(par)
#define SC_WINDOW(xb,xe,yb,ye)  OUTB(WINDOW); OUTDW(xb,0);OUTDW(xe,0); \
                                OUTDW(SC__Y(ye),0);\
                                OUTDW(SC__Y(yb),0);
#define SC_VWPORT(xb,xe,yb,ye)  OUTB(VWPORT); OUTW(xb);OUTW(xe); \
                                OUTW(SC__Y(ye));OUTW(SC__Y(yb));
#ifdef VAX
#define SC_XMOVE(x,y)           OUTB(XMOVE); OUTW(x);OUTW(SC__Y(y));\
                                SC_FLUSH()
#define SC_XHAIR(par,size_x,size_y) OUTB(XHAIR); OUTB(par); \
                                OUTW(size_x);OUTW(size_y)
#define SC_DISABLE_XHAIR        OUTB(XHAIR); OUTB(0)
#define SC_RBAND(par)           OUTB(RBAND);OUTB(par)
#define SC_TEXTC(size,str)      OUTB(TEXTC);OUTW(size);OUTS(str)
#define SC_RASTOP(oper,sdir,ddir,x0,x1,y0,y1,x2,y2)        \
                                OUTB(RASTOP);OUTB(oper);OUTB(sdir);OUTB(ddir); \
                                OUTW(x0);OUTW(x1);OUTW(y0);OUTW(y1);OUTW(x2);OUTW(y2)
#endif

#define SC_CLEAR_BLOCK(color,x_b,x_e,y_b,y_e)  \
                                SC_VWPORT(x_b,x_e,y_b,y_e); \
                                SC_FLOOD(color); \
                                SC_VWPORT(0,SC_X_MAX-1,0,SC_Y_MAX-1)

#endif /* SC_QG */

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
extern unsigned long SCX_color_translation[SC_C_FULL];

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


#ifdef SC_PC
#define FAR
extern unsigned FAR SC__curPointX, FAR SC__curPointY, FAR SC__filled;
extern unsigned FAR zmax,FAR xmax, FAR ymax;
extern int FAR color;
extern unsigned int FAR textcols;
extern unsigned int FAR textrows;
extern unsigned int FAR text_height;
extern int FAR vmode;
extern int FAR set_color(int);
extern void FAR box(int,int,int,int);
extern void FAR filled_box(int,int,int,int);
extern void FAR line(int,int,int,int);
extern void FAR ellipse(int,int,int,unsigned int);
extern void FAR color_s(int x,int y,int color,char FAR *str);
extern void (FAR *block)(image_t *buffer,int x,int y,int sizeX,int sizeY);
extern void (FAR *dot)(int x,int y);
extern void FAR getmode(void);
extern void FAR setmode(int mode);
extern void FAR init(void);
extern void FAR vidmode(int);

#define SC_START()      { getmode(); init(); \
			  SC_COLOR(SC_C_ANNOTATE);\
			  SC_MASK(SC_M_ALL);\
			}
#define SC_START_BIG()  { getmode(); init(); \
			  SC_COLOR(SC_C_ANNOTATE);\
			  SC_MASK(SC_M_ALL);\
			}
#define SC_STOP()       { getchar();vidmode(3); }
#define SC_STOP_CLEAR() vidmode(3)
#define SC_FLUSH()
#define SC_PutImg(image,x,y,lx,ly) (*block)(image,x,y,(lx),(ly))
#define SC_X_MAX /*xmax*/640
#define SC_Y_MAX /*ymax*/350
#define SC_CHAR_WIDTH   10
#define SC_CHAR_HEIGHT  10
/* There is a problem here: you don't know the sizes of the screen
   until you called getmode(). This is currently done with the
   SC_START() macro, which immediately calls init() afterwards.
   If you put SC_X_MAX equal to xmax, this will contain only sensible
   values after SC_START().
*/

#define SC_MASK(mask)
#define SC_TJUST(hor,ver)
#define SC_MOVE(x,y)  {SC__curPointX = (unsigned FAR)(x);\
			SC__curPointY = (unsigned FAR)(y);}
#define SC_POINT()    (*dot)(SC__curPointX, SC__curPointY)
#define SC_RECT(x,y)  (SC__filled\
                       ? filled_box(SC__curPointX,SC__curPointY,x,y)\
                       : box(SC__curPointX,SC__curPointY,x,y))
#define SC_RECTR(x,y)  (SC__filled\
                       ? filled_box(SC__curPointX,SC__curPointY,\
			 SC__curPointX+x,SC__curPointY+y)\
                       : box(SC__curPointX,SC__curPointY,\
			 SC__curPointX+x,SC__curPointY+y))
#define SC_CLEAR_BLOCK(col,x_b,x_e,y_b,y_e)  \
                      { set_color(col);\
                        filled_box(x_b,y_b,x_e,y_e);\
                        set_color(color);\
                      }
#define SC_ELLIPSE(x,y) ellipse(SC__curPointX,SC__curPointY,\
			x,(unsigned)((y*256L)/x))
#define SC_CIRCLE(x)  SC_ELLIPSE(x,x)
#define SC_DRAW(x,y)  {line(SC__curPointX,SC__curPointY,\
			(unsigned)(x),(unsigned)(y)); SC_MOVE(x,y);}
#define SC_LINE(x1,y1,x2,y2) \
			line((unsigned)x1, (unsigned)y1, \
				SC__curPointX = (unsigned)(x2),\
				SC__curPointY = (unsigned)(y2))
#define SC_COLOR(col) set_color(col)
#define SC_PRMFIL(fil) (SC__filled = fil)
#define SC_TEXT(str)  color_s((SC__curPointY)/text_height,SC__curPointX/8,color,str);

#define SC_C_ANNOTATE   (zmax-1)
#define SC_C_BACKGROUND 0
#define SC_C_MAX	(zmax-1)       	/* maximum color to be used */
#define SC_C_FULL	(zmax-1)	/* maximum color available */
/* two definitions for masking availability */
#define SC_M_ANNOTATE   0x80
#define SC_M_ALL        0xff
#endif /* SC_PC */


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
