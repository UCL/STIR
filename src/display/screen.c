/* 
 $Id$
*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, IRSL
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
#define SCreen_compiling 1
#include "screen.h"

#ifdef SC_QG

#ifdef VAX
#include descrip
#include iodef
#include dcdef
#include ssdef

static int channel;
static long iosb[2],mode1[2],mode2[2];
static $DESCRIPTOR(PGAdevice_dsc,PGA);

void error_check(sys_function)
int sys_function;
{ int status;

  if ((status=sys_function) !=1)
  { lib$signal(status);
    exit(EXIT_ABORT);
  }
}

static openpga()
{
/*  if ((outfile=fopen(PGA,"w","rfm=fix","mrs=2035"))==0)
  {  perror("Error opening PGA");
     return(1);
  }
  else
     return(0);
*/
  error_check(sys$assign(&PGAdevice_dsc,&channel,0,0));
  error_check(sys$qiow(0,channel,IO$_SENSEMODE,&mode1,0,0,0,0,0,0,0,0));
  memcpy(mode2, mode1, sizeof(mode1));
  mode2[0] = (132 << 16) | DC$_LP;
  /*mode2[1] |= LP$M_PASSALL;*/
  error_check(sys$qiow(0,channel,IO$_SETMODE,&iosb,0,0,mode2,
                sizeof(mode2),0,0,0,0));
}

static closepga()
{    error_check(sys$dassgn(channel));}

#define MAX_LENGTH_BUFFER 2035

static unsigned output(buffer,size)
out_t    *buffer;
unsigned size;
{ unsigned i;

  for (i=0;i<size;i+=MAX_LENGTH_BUFFER)
  sys$qiow(0,channel,IO$_WRITEVBLK,&iosb,0,0,
          buffer+i,(MAX_LENGTH_BUFFER<size-i ? MAX_LENGTH_BUFFER : size-i),0,0,0,0);
}
#endif
#ifdef MSDOS

static FILE *outfile;
#include <dos.h>
static openpga()
{ union REGS inregs,outregs;
  int handle;
  unsigned char dev_info;

  if ((outfile=fopen(PGA,"wb"))==NULL)
  {  perror("Error opening PGA");
     exit(EXIT_ABORT);
  }
  handle = fileno(outfile);
  inregs.h.ah= 0x44;                        /*IOCTL*/
  inregs.h.al= 0x00;                        /*get device info*/
  inregs.x.bx= handle;
  intdos(&inregs, &outregs);
  if ( ((dev_info = outregs.h.dl) & 0x80) == 0)
    if (!ask("PGA is not a device !! Continue",'Y'))
      exit(EXIT_ABORT);

  inregs.h.ah= 0x44;                        /*IOCTL*/
  inregs.h.al= 0x01;                        /*set device info*/
  inregs.x.bx= handle;
  inregs.h.dh= 0;
  inregs.h.dl= dev_info | 0x20;             /*BINARY device*/
  intdos(&inregs, &outregs);
}

static closepga()
{    fclose(outfile);}

static unsigned output(buffer,size)
out_t    *buffer;
unsigned size;
{
  fwrite(buffer,size,1,outfile);
}
#endif

SC_INIT(hex)
int hex;
{
     outptr=buffer;
     openpga();
     OUTB('C');
     if (hex==HEX)
       OUTB('X');
     else
       OUTB('A');
     OUTB(' ');
     if (hex==HEX)
     { SC_VWPORT(0,SC_X_MAX-1,0,SC_Y_MAX-1);
       SC_WINDOW(0,SC_X_MAX-1,0,SC_Y_MAX-1);
       SC_MASK(SC_M_ALL);
       SC_COLOR(SC_C_ANNOTATE);
     }
     return 0;
}
static int  write_test=0;
static FILE *test;

SC_INIT_FILE(hex,name)
int hex;
char name[];
{    write_test=1;
#ifdef VAX
     if ( (test=fopen(name,"w","rfm=fix","mrs=1024"))==0)
#else
     if ( (test=fopen(name,"wb"))==0)
#endif
     {  perror("Error opening testfile");
        return(EXIT_ABORT);
     }
     else
        return SC_INIT(hex);
}

SC_FLUSH()
{ unsigned size;
     output(buffer,size=(outptr-buffer));
#ifndef VAX
     fflush(outfile);
#endif
     if (write_test && size!=0)
       fwrite(buffer,size,1,test);
     outptr = buffer;
}

SC_CLOSE(hex)
int hex;
{
     OUTB('C');
     if (hex==HEX)
        OUTB('X');
     else
        OUTB('A');
     OUTB(' ');
     SC_FLUSH();
     if (write_test)
         fclose(test);
     closepga();
}


static void output_equal  (image_t *, image_t *);
static void output_diff   (image_t *, image_t *);
static void no_rl_code    (unsigned int, image_t *);
static void rl_code       (unsigned int, image_t *);

int SCQ_rl;

void SCQ_PutImg (image,x_begin,y_begin,x_length,y_length)
  image_t *image;
  int x_begin,y_begin, x_length,y_length;
{
 short int x_end,y_end,y;
 register  y_temp;

 x_end   = x_begin+x_length-1;
 y_end   = y_begin+y_length-1;
 SC_MASK(SC_M_ALL);
 SC_CLEAR_BLOCK(SC_C_BACKGROUND,x_begin,x_end,y_begin,y_end);

 for (y_temp=0,y=y_begin;y_temp<y_length;y_temp++,y++, image+= x_length)
 {
  SC_IMAGEW(y, x_begin,x_end);
  if (SCQ_rl)
    rl_code (x_length,image);
  else
    no_rl_code (x_length,image);

#ifdef MSDOS
  SC_FLUSH();
#endif
 }
#ifdef VAX
  SC_FLUSH();
#endif
}

static void no_rl_code (length,delta)
image_t *delta;
unsigned int length;
{ register j;
  int quot,rest;

  quot= length/128;
  rest= length%128;
  for (j=0; j<=quot-1; j++)
  { OUTB(127 | 0x80);
    OUTM(delta+j*128,128);
  }
  OUTB(rest | 0x80);
  OUTM(delta+quot*128,rest+1);
}

static void rl_code (length,delta)
image_t *delta;
unsigned int length;
{
 register image_t *cur,*start;
 int diff,i;

 for (cur=start=delta, i=length; i>0; i--, cur++)
 { if (cur==start)
     diff= *start!=*(start+1);
   else
     if (diff && (*cur==*(cur+1)))
     { output_diff(start,cur);
       start= cur;
       diff=0;
     }
     else if (!diff && (*cur!=*(cur+1)))
     { output_equal(start,cur);
       start= cur+1;
     }
 }
 if (*start==*cur)
   output_equal(start,cur);
 else
   output_diff(start,cur+1);
}

static void output_equal(start,cur)
image_t *start,*cur;
{ int quot,rest,tmp;
  register i;

 quot= (tmp=cur-start) / RLCMAX;
 rest=tmp % RLCMAX;
 for (i=quot; i>0; i--)
 {  OUTB(RLCMAX - 1);
    OUTB( *start);
 }
 OUTB(rest);
 OUTB(*start);
}

static void output_diff(start,cur)
image_t *start,*cur;
{ int quot,rest,tmp;
  register i;

 quot= (tmp=cur-start-1) / RLCMAX;
 rest=tmp % RLCMAX;
 for (i=quot; i>0; i--)
 {  OUTB(255);
    OUTM(start,RLCMAX);
    start+= RLCMAX;
 }
 OUTB(rest + RLCMAX) ;
 OUTM(start,rest+1);
}


#ifdef VAX

void write_buffer(device,buffer)
char *buffer,*device;
{
int channel;
long iosb[2];
long mode1[2];
long mode2[2];
struct
{ long length;
  char *pointer;
} devnam;

devnam.pointer=device;
devnam.length=strlen(device);

error_check(sys$assign(&devnam,&channel,0,0));
error_check(sys$qiow(0,channel,IO$_SENSEMODE,&mode1,0,0,0,0,0,0,0,0));
memcpy(mode2, mode1, sizeof(mode1));
mode2[0] = (132 << 16) | DC$_LP;
/*mode2[1] |= LP$M_PASSALL;*/
error_check(sys$qiow(0,channel,IO$_SETMODE,&iosb,0,0,mode2,
                sizeof(mode2),0,0,0,0));
error_check(sys$qiow(0,channel,IO$_WRITEVBLK,&iosb,0,0,
                buffer,strlen(buffer),0,0,0,0));
error_check(sys$dassgn(channel));
}


SC_pixel_t *read_buffer(device,size)
char *device;
int size;
{
#define PORT_FULL       8
#define MAX_LENGTH_BUFFER 2036
unsigned int channel,i,qg_status;
SC_pixel_t *buffer;
long iosb[2];
long mode1[2];
long mode2[2];
struct
{ long length;
  char *pointer;
} devnam;

devnam.pointer=device;
devnam.length=strlen(device);

error_check(sys$assign(&devnam,&channel,0,0));
/* KT 12/01/2000 replace malloc_check by explicit checking */
if ((buffer=malloc(size))==NULL)
    error("Reading image for display: error allocating %d bytes",size);


error_check(sys$qiow(0,channel,IO$_SENSEMODE,&mode1,0,0,0,0,0,0,0,0));
memcpy(mode2, mode1, sizeof(mode1));
mode2[0] = (132 << 16) | DC$_LP;
/*mode2[1] |= LP$M_PASSALL;*/
error_check(sys$qiow(0,channel,IO$_SETMODE,&iosb,0,0,mode2,
                sizeof(mode2),0,0,0,0));
for (i=0;i<size;i+=MAX_LENGTH_BUFFER)
 sys$qiow(0,channel,IO$_READVBLK,&iosb,0,0,
         buffer+i,(MAX_LENGTH_BUFFER<size-i ? MAX_LENGTH_BUFFER :
size-i),0,0,0,0);
qg_status=iosb[1]&0xff;
if (qg_status & PORT_FULL)
  error
  ("Error to read out the Data port register of the QG-640 card \
completely!");
error_check(sys$dassgn(channel));
return (buffer);
}
#endif

#endif /* SC_QG */

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
unsigned long SCX_color_translation[SC_C_FULL];

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
	printf("%d:  %s",i,all_color_scales[i].name);
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
	CurrentColormap = ++CurrentColormap % NUMBER_SCALES;

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
			  SCX_pixmap_format.bits_per_pixel, bytes_per_line);
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

#ifdef SC_PC
#include <dos.h>
#undef peek
#undef pokeb
#undef pokew

#define EGA      0
#define CHIPS435 1
#define CHIPS441 2
#define CIRRUS   3
#define TSENG    4
#define VGA      5
#define VGAWONDER 6
int ATI_REG;
void WONDER_plane(int p);
#define MAX_ROW 768
/***********************************************************************
 *  Function call explanations:
 *
 *  (*color_c)(row, col, attribute, character)
 *      writes the character "character" on text row "row", character
 *      column "col" using the color specified in the low byte of
 *      "attribute".  If bit 8 of "attribute" is a 1, the background
 *      color is undisturbed, otherwise it is set to color 0 (Black).
 *      If bit 9 of "attribute" is 1, the character is exclusived-ORed
 *      with current contents of the character cell rather than normal
 *      "replace" option.  If both bits 8 and 9 are set to 1, the operation
 *      is UNDEFINED and in text modes bits 8 and 9 are ignored.
 *
 ***********************************************************************/

int ega_bit(), ega_get(), egacolor();
extern void ega_block(char huge *buffer,int x,int y,int sizeX,int sizeY);
int one_bit(), one_get(), onecolor();
int two_bit(), two_get(), twocolor();
int four_bit(), four_get(), fourcolor();
void dummy(char huge *buffer,int x,int y,int sizeX,int sizeY)
{
}
unsigned char far *getfont();
unsigned int rr();

int chiptype;
unsigned int regen;
unsigned int memend;
unsigned int mode_reg;
unsigned int color_reg;
unsigned int row_start[MAX_ROW];   /* start address of each scan row */
unsigned char bank_sel[MAX_ROW];   /* page used for each scan row */
unsigned int expand_left[256];  /* graphics text char pattern expand tables */
unsigned int expand_right[256]; /*   that expand 1 bit per pixel into 2 or 4 bpp */
unsigned char far *fontptr;     /* character generator table base address */
unsigned int text_height = 8;   /* graphics char height */
int (*color_c)();               /* the current write char */
unsigned int clear_word;        /* word value to clear the buffer with */
unsigned int aspect;            /* aspect ratio */

unsigned int xmax;
unsigned int ymax;
unsigned int zmax;
unsigned int bits_per_pixel;
unsigned int pixels_per_byte;
unsigned int textcols;
unsigned int textrows;
int digital_flag;
unsigned int limit1;
unsigned int SC__curPointX,SC__curPointY,SC__filled;

void (*dot)(int,int);           /* the current pixel output routine */
unsigned int (*get_dot)();      /* the current read  pixel routine */
int (*find_color)();            /* the current find pixel of color routine */
void (*block)(image_t *buffer,int x,int y,int sizeX,int sizeY);

int vmode;
int color;                      /* current drawing color */
unsigned int *color_table;      /* pointer to current color expand table */
unsigned char *get_shift;       /* pointer to read pixel shift count table */

/* tables for one bit per pixel */
unsigned char ega_mask[] =   { 0x80, 0x40, 0x20, 0x10, 0x08, 0x04, 0x02, 0x01 };
unsigned int  ega_color[] =  { /* EGA 16-color modes expand a color to itself */
   0x0000, 0x0101, 0x0202, 0x0303, 0x0404,
   0x0505, 0x0606, 0x0707, 0x0808, 0x0909,
   0x0A0A, 0x0B0B, 0x0C0C, 0x0D0D, 0x0E0E, 0x0F0F};
unsigned int  ega_mono_color[] =  { /* EGA mono mode */
   0x0000, 0x0101, 0x0505, 0x0404};
 /* black, video,  intens, blink */
unsigned char one_mask[] =   { 0x7f, 0xbf, 0xdf, 0xef, 0xf7, 0xfb, 0xfd, 0xfe };
unsigned int  one_color[] =  { /* color pattern expand table */
   0x0000, 0xFFFF, 0xFFFF, 0xFFFF,
   0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF,
   0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF,
   0xFFFF, 0xFFFF, 0xFFFF, 0xFFFF
};
char one_shift[] =  { 7, 6, 5, 4, 3, 2, 1, 0 };

/* tables for two bits per pixel */
unsigned char two_mask[] =   { 0x3F, 0xCF, 0xF3, 0xFC };
unsigned int  two_color[] =  {
   0x0000, 0x5555, 0xAAAA, 0xFFFF,
   0xFFFF, 0x5555, 0xAAAA, 0xFFFF,
   0xFFFF, 0x5555, 0xAAAA, 0xFFFF,
   0xFFFF, 0x5555, 0xAAAA, 0xFFFF
};
char two_shift[] =  { 6, 4, 2, 0 };

/* tables for four bits per pixel */
unsigned char four_mask[] =   { 0x0F, 0xF0 };
unsigned int  four_color[] =  {
   0x0000, 0x1111, 0x2222, 0x3333,
   0x4444, 0x5555, 0x6666, 0x7777,
   0x8888, 0x9999, 0xAAAA, 0xBBBB,
   0xCCCC, 0xDDDD, 0xEEEE, 0xFFFF
};
char four_shift[] =  { 4, 0 };

typedef struct {
   int mode;
   int pixels;
   int lines;
   unsigned char chars;
   unsigned char rows;
   int colors;
   unsigned int segmt;
   unsigned char charsize;
   unsigned char text;
   unsigned int size;
   unsigned char frames;
   unsigned char bits;
   unsigned int addr;
   unsigned char blink;
   } mode_t;
mode_t mode_tbl[] = {           /* only graph modes */
      {  4, 320, 200,  40, 25,  4, 0xB800,  8, 0, 0x4000, 2, 2, 0x3D4, 0},
      {  5, 320, 200,  40, 25,  4, 0xB800,  8, 0, 0x4000, 2, 2, 0x3D4, 0},
      {  6, 640, 200,  80, 25,  2, 0xB800,  8, 0, 0x4000, 2, 1, 0x3D4, 0},
      {  9, 720, 348,  90, 43,  2, 0xB000,  8, 0, 0x8000, 4, 1, 0x3B4, 0},
      { 11, 704, 519,  88, 64, 16, 0xA000,  8, 0, 0xC000, 1, 1, 0x3D4, 1},
      { 12, 832, 350, 104, 25, 16, 0xA000, 14, 0, 0xA000, 1, 1, 0x3D4, 0},
      { 13, 320, 200,  40, 25, 16, 0xA000,  8, 0, 0x2000, 1, 1, 0x3D4, 0},
      { 14, 640, 200,  80, 25, 16, 0xA000,  8, 0, 0x4000, 1, 1, 0x3D4, 0},
      { 15, 640, 350,  80, 25,  4, 0xA000, 14, 0, 0x8000, 1, 1, 0x3B4, 1},
      { 16, 640, 350,  80, 25, 16, 0xA000, 14, 0, 0x8000, 1, 1, 0x3D4, 0},
      { 17, 640, 480,  80, 30,  2, 0xA000, 16, 0, 0xA000, 1, 1, 0x3D4, 0},
      { 18, 640, 480,  80, 30, 16, 0xA000, 16, 0, 0xA000, 1, 1, 0x3D4, 0},
      { 19, 320, 200,  40, 25,256, 0xA000,  8, 0,  64000, 1, 8, 0x3D4, 0},
      { 25, 720, 348,  90, 43,  2, 0xB000,  8, 0, 0x8000, 4, 1, 0x3B4, 0},
      { 37, 640, 480,  80, 30, 16, 0xA000, 16, 0, 0xA000, 1, 1, 0x3D4, 0},
      { 41, 800, 600, 100, 75, 16, 0xA000,  8, 0,  60416, 1, 1, 0x3D4, 0},
      { 45, 640, 350,  80, 25,256, 0xA000, 14, 0, 0xFFFF, 1, 8, 0x3D4, 0},
      { 46, 640, 480,  80, 30,256, 0xA000, 16, 0, 0xFFFF, 1, 8, 0x3D4, 0},
      { 48, 800, 600, 100, 75,256, 0xA000,  8, 0, 0xFFFF, 1, 8, 0x3D4, 0},
      { 54, 960, 720, 120, 45, 16, 0xA000, 16, 0, 0xFFFF, 1, 1, 0x3D4, 1},
      { 55,1024, 768, 128, 48, 16, 0xA000, 16, 0, 0xFFFF, 1, 1, 0x3D4, 1},
      { 0x54, 800, 600, 100, 75, 16, 0xA000,  8, 0,  60416, 1, 1, 0x3D4, 0},
      { 0x62, 640, 480,  80, 30,256, 0xA000, 16, 0, 0xFFFF, 1, 8, 0x3D4, 0},
      { 0x63, 800, 600, 100, 75,256, 0xA000,  8, 0, 0xFFFF, 1, 8, 0x3D4, 0},
      { 0x65,1024, 768, 128, 48, 16, 0xA000, 16, 0, 0xFFFF, 1, 4, 0x3D4, 1},
      { 0,    0,   0,   0,  0,  0,      0,  0, 0,      0, 0, 0,     0, 0}
      };


/**************************** start rainbow ****************************/
rainbow()
{ int rainbow[256], offset,i,j;
  char pal_tab[257*3];            /* Maximum size */
#define length 255
  for (i=0; i < 256; i++)
    rainbow[i] = 0;
  sine_tab(/*129*/length, 63, rainbow);
  offset = length/3;
  offset = offset * 3 + 1;
  for (j = 0, i = 0; i <= length; j += 3, i++)
  { pal_tab[j] = rainbow[i];
    pal_tab[(j+offset)%(length*3)] = rainbow[i];
    pal_tab[(j+offset*2)%(length*3)] = rainbow[i];
  }
  set_palette(1, length, &pal_tab[0]);
#undef length
}

grey_scale()
{ int i;
  char pal_tab[257*3];            /* Maximum size */

  for (i = 0; i < 256; i++)
  { pal_tab[3*i] = pal_tab[3*i+1] = pal_tab[3*i+2] = (char)(i>>2);
  }
  set_palette(0, 256, &pal_tab[0]);
}

sine_tab(len, max, table)
   int len, max;
   int *table;
   {
   int col;
   long two_a, two_b, d;
   long a, b, aincr, bincr;
   int center, adj;

   col = 0;
   center = len/2;
   a = ((long)len)*((long)len);
   two_a = a+a;
   b = ((long)max)*((long)max);
   two_b = b+b;
   d = 210L * max;
   aincr = -a*d/len;
   bincr = b*d/len;
   d = ((b - a) << 1) + (a >> 1);

   while (bincr + aincr < 0 && col <= center) {
      table[center+col] = max;
      table[center-col] = max;
      d += (bincr += two_b);
      col++;
      if (d >= 0) {
         d += (aincr += two_a);
         max--;
         }
      }
   while (col <= center) {
      table[center+col] = max;
      table[center-col] = max;
      if (d <= 0) {
         d += (bincr += two_b);
         col++;
         }
      max--;
      d += (aincr += two_a);
      }
   }
/**************************** end rainbow ******************************/
/*************************** start getmode *****************************/

void getmode()
{ int im,is8514,col256;
  int i;

  is8514 = 1;
  if (inp(0x3C2) & 0x60)
    is8514 = 0;
  chiptype = chipset();

  i = peek(0x488,0) & 0x0F;
  digital_flag = 1;
  if (col256 = color256())
    digital_flag = 0;

  if (peek(0x463,0) == 0x3B4)                           /* Monochrome */
  { vmode = 9;                  /* 720x348 (Hercules) monochrome graphics */
    if (chiptype == TSENG || chiptype == EGA || chiptype == VGA)
      vmode = 15;               /* 640x350 (EGA) Monochrome graphics */
  }
  else if (i != 3 && i != 9)                            /* CGA */
    vmode = 14;                 /* 640 x 200 16-color graphics */
  else if ((peek(0x489,0) & 0x10) == 0)                 /* EGA */
  { vmode = 16;                 /* 640 x 350 16-color graphics */
    if (col256 && chiptype == TSENG)
      vmode = 45;               /* 640 x 350 64-color graphics */
  }
  else if (is8514 && (chiptype == TSENG))
  { /*vmode = 18;*/             /* 640 x 480 16-color graphics */
    vmode = 55;                 /* 1024 x 768 16-color graphics */
    if (col256)
      vmode = 46;               /* 640 x 480 256-color graphics */
  }
  else if (is8514 && (chiptype == VGAWONDER))
  { /*vmode = 18;*/             /* 640 x 480 16-color graphics */
    vmode = 0x65;               /* 1024 x 768 16-color graphics */
    if (col256)
      vmode = 0x62;             /* 640 x 480 256-color graphics */
  }
  else if (is8514 || (read_emul() & 0x80))              /* VGA */
  { vmode = 18;                 /* 640 x 480 16-color graphics */
    if (col256)
      if (chiptype == TSENG)
        vmode = 46;             /* 640 x 480 256-color graphics */
      else if (chiptype == VGAWONDER)
        vmode = 0x62;           /* 640 x 480 256-color graphics */
  }
  else                                                  /* Multisync */
  { vmode = 18;                 /* 640 x 480 16-color graphics */
    if (chiptype == TSENG)
    { vmode = 41;               /* 800 x 600 16-color graphics */
      if (col256)
        vmode = 46;             /* 640 x 480 256-color graphics */
    }
    else if (chiptype == VGAWONDER)
    { vmode = 0x54;             /* 800 x 600 16-color graphics */
      if (col256)
        vmode = 0x62;           /* 640 x 480 256-color graphics */
        /*vmode = 0x63;*/       /* 800 x 600 256-color graphics */
    }
  }
  setmode(vmode);
}

chipset()
   {
   int temp;
   int type;
   int crtcaddr;

   crtcaddr = peek(0x463,0);

/* start Kris */
   if ((char)peek(0x31,0xc000) != '7') goto other;
   if ((char)peek(0x32,0xc000) != '6') goto other;
   if ((char)peek(0x33,0xc000) != '1') goto other;
   if ((char)peek(0x34,0xc000) != '2') goto other;
   if ((char)peek(0x35,0xc000) != '9') goto other;
   if ((char)peek(0x36,0xc000) != '5') goto other;
   if ((char)peek(0x37,0xc000) != '5') goto other;
   if ((char)peek(0x38,0xc000) != '2') goto other;
   if ((char)peek(0x39,0xc000) != '0') goto other;
   /* ATI product */
   if ((char)peek(0x40,0xc000) != '3') goto other;
   if ((char)peek(0x41,0xc000) != '1') goto other;
   type = VGAWONDER;
   ATI_REG = peek(0x10,0xc000);
   return(type);
other:
/* end Kris */
   type = EGA;
   temp = rr(crtcaddr, 0xFF);
   outp(crtcaddr+1, 0xC0 ^ temp);
   if (((rr(crtcaddr, 0xFF)^temp) ^ 0xC0) == 0 && rr(crtcaddr, 0xF7) == 0 &&
         rr(crtcaddr, 0xF8) == 0) {
      outp(crtcaddr, 0xFF);
      outp(crtcaddr+1, temp);
      temp = rr(0x3CE, 0xF7);
      outp(0x3CF, 0xFF ^ temp);
      if ((temp ^ rr(0x3CE, 0xF7) ^ 0xFF) == 0) {
         outp(0x3CF, temp);
         type = CHIPS441;
         }
      else {
         outp(0x3CF, temp);
         type = CHIPS435;
         }
      }
   else {
      outp(crtcaddr, 0xFF);
      outp(crtcaddr+1, temp);
      temp = rr(crtcaddr, 0x16);
      outp(crtcaddr+1, 0xFF ^ temp);
      if ((rr(crtcaddr, 0x16) ^ temp ^ 0xFF) == 0) {
         outp(crtcaddr, 0x16);
         outp(crtcaddr+1, temp);
         if (((temp=rr(crtcaddr, 0x23)) & 0x03) == 0) {
            outp(crtcaddr+1, 0x83 | temp);
            if ((rr(crtcaddr, 0x23) & 0x83) == 0x83) {
               outp(crtcaddr, 0x23);
               outp(crtcaddr+1, temp);
               type = TSENG;
               }
            else {
               outp(crtcaddr, 0x23);
               outp(crtcaddr+1, temp);
               goto ibm;
               }
            }
         else
            ibm:
            type = VGA;
         }
      }
   if ((rr(crtcaddr, 0x7F) ^ rr(crtcaddr, 0x0C) ^ 0xEA) == 0)
      return (CIRRUS);
   return (type);
   }

static unsigned int rr(port, reg)  /* only used in chipset */
   int port;
   int reg;
   {
   outp(port, reg);
   return (inp(port+1) & 0xFF);
   }

static int color256() /* only used in getmode */
   {
   outp(0x3C6, 0x55);
   if ((inp(0x3C6) & 0xFF) != 0x55)
      return (0);
   outp(0x3C6, 0xAA);
   if ((inp(0x3C6) & 0xFF) != 0xAA)
      return (0);
   outp(0x3C6, 0x00);
   if ((inp(0x3C6) & 0xFF) != 0x00)
      return (0);
   outp(0x3C6, 0xFF);
   if ((inp(0x3C6) & 0xFF) != 0xFF)
      return (0);
   return (1);
   }
/***************************** end getmode *****************************/

one_char(row, col, attr, c)
   unsigned int row, col;
   unsigned int attr;
   unsigned int c;
   {
   unsigned i, seg, ofs, bkcolor;
   unsigned char far *ptr;

   ptr = fontptr + c * text_height;
   row *= text_height;
   if (zmax < 4) {
      if (attr & 0x0100) {
         if (attr & 0x00FF)
            for(i=0; i<text_height; i++) {
               bkcolor = peek(row_start[row]+col, regen);
               pokeb(row_start[row++]+col, regen, *ptr++ | bkcolor);
               }
         else
            for(i=0; i<text_height; i++) {
               bkcolor = peek(row_start[row]+col, regen);
               pokeb(row_start[row++]+col, regen, (~*ptr++) & bkcolor);
               }
         }
      else if (attr & 0x0200)
         for(i=0; i<text_height; i++) {
            bkcolor = peek(row_start[row]+col, regen);
            pokeb(row_start[row++]+col, regen, *ptr++ ^ bkcolor);
            }
      else
         for(i=0; i<text_height; i++) {
            pokeb(row_start[row++]+col, regen, *ptr++);
            }
      }
   else {
      (*get_dot)(0, row);               /* Initialize "regen" */
      if ((attr & 0x0200) != 0) {
         wrt_ega(((attr<<8) & 0x0F00) | 0x02, 0x3C4);
         wrt_ega(0x1803, 0x3CE);
         for(i=0; i<text_height; i++)
            pokeb(row_start[row++]+col, regen, *ptr++);
         wrt_ega(0x0003, 0x3CE);
         wrt_ega(0x0F02, 0x3C4);
         return;
         }
      wrt_ega(0x0205, 0x3CE);
      if ((attr & 0x0100) == 0) {
         wrt_ega(0xFF08, 0x3CE);
         bkcolor = (attr >> 4) & 0x0F;
         for (i=0; i<text_height; i++)
            pokeb(row_start[row++]+col, regen, bkcolor);
         row -= text_height;
         }
      for(i=0; i<text_height; i++) {
         wrt_ega((*ptr++ << 8) + 8, 0x3CE);
         pokeb(row_start[row++]+col, regen, attr);
         }
      wrt_ega(0xFF08, 0x3CE);
      wrt_ega(0x0005, 0x3CE);
      }
   }

two_char(row, col, attr, c)
   unsigned int row, col;
   unsigned int attr;
   unsigned int c;
   {
   unsigned i, bkcolor;
   unsigned char far *ptr;

   ptr = fontptr + c * text_height;
   col <<= 1;
   row *= text_height;
   if (attr & 0x0100)
      for(i=0; i<text_height; i++) {
         bkcolor = peek(row_start[row]+col, regen)
               & (~expand_left[*ptr & 0xFF]);
         pokew(row_start[row++]+col, regen, (expand_left[*ptr++]
               & color_table[attr & 0x0F]) | bkcolor);
         }
   else if (attr & 0x0200)
      for(i=0; i<text_height; i++) {
         bkcolor = peek(row_start[row]+col, regen)
               & (~expand_left[*ptr & 0xFF]);
         pokew(row_start[row++]+col, regen, (expand_left[*ptr++]
               & color_table[attr & 0x0F]) ^ bkcolor);
         }
   else
      for(i=0; i<text_height; i++) {
         pokew(row_start[row++]+col, regen, expand_left[*ptr++]
               & color_table[attr & 0x0F]);
         }
   }

four_char(row, col, attr, c)
   unsigned int row, col;
   unsigned int attr;
   unsigned int c;
   {
   unsigned int i, bkcolor, previous,test;
   unsigned char far *ptr;

   ptr = fontptr + c * text_height;
   col <<= 2;
   row *= text_height;
   if (chiptype == VGAWONDER) {
      WONDER_plane(previous = bank_sel[row]);
      test = row_start[row];
      }
   if (attr & 0x0100)
      for(i=0; i<text_height; i++) {
         bkcolor = peek(row_start[row]+col, regen) &
               (~expand_left[c = *ptr++]
               & color_table[attr & 0x0F]);
         if (chiptype == VGAWONDER && test > row_start[row]+col/2) {
            test = row_start[row]+col;
            WONDER_plane(++previous);
            }
         pokew(row_start[row]+col, regen, (expand_left[c]
               & color_table[attr & 0x0F]) | bkcolor);
         bkcolor = peek(row_start[row]+col+2, regen) & (~expand_right[c]);
         if (chiptype == VGAWONDER && test > row_start[row]+col/2+1) {
            test = row_start[row]+col+2;
            WONDER_plane(++previous);
            }
         pokew(row_start[row++]+col+2, regen, (expand_right[c]
               & color_table[attr & 0x0F]) | bkcolor);
         }
   else if (attr & 0x0200)
      for(i=0; i<text_height; i++) {
         bkcolor = peek(row_start[row]+col, regen) &
               (~expand_left[c = *ptr++]
               & color_table[attr & 0x0F]);
          if (chiptype == VGAWONDER && ((test ^ row_start[row]+col/2) & 0x8000)) {
             test = row_start[row]+col;
             WONDER_plane(++previous);
             }
         pokew(row_start[row]+col, regen, (expand_left[c]
               & color_table[attr & 0x0F]) | bkcolor);
         bkcolor = peek(row_start[row]+col+2, regen) & (~expand_right[c]);
          if (chiptype == VGAWONDER && ((test ^ row_start[row]+col/2+1) & 0x8000)) {
             test = row_start[row]+col+2;
             WONDER_plane(++previous);
             }
         pokew(row_start[row++]+col+2, regen, (expand_right[c]
               & color_table[attr & 0x0F]) ^ bkcolor);
         }
   else
      for(i=0; i<text_height; i++) {
         if (chiptype == VGAWONDER && ((test ^ row_start[row]+col/2) & 0x8000)) {
            test = row_start[row]+col/2;
            WONDER_plane(++previous);
            }
         pokew(row_start[row]+col, regen, expand_left[c = *ptr++]
               & color_table[attr & 0x0F]);
         if (chiptype == VGAWONDER && ((test ^ row_start[row]+col/2+1) & 0x8000)) {
            test = row_start[row]+col/2+1;
            WONDER_plane(++previous);
            }
         pokew(row_start[row++]+col+2, regen, expand_right[c]
               & color_table[attr & 0x0F]);
         }
   }

eight_char(row, col, attr, c)
   unsigned int row, col, attr, c;
   {
   unsigned int i, j, k;
   int bkcolor;
   unsigned char far *ptr;
   unsigned int test,previous;

   ptr = fontptr + c * text_height;
   col <<= 3;
   row *= text_height;
   if (chiptype == TSENG) {
      outp(0x3CD, bank_sel[row]);
      test = row_start[row];
      }
   if (chiptype == VGAWONDER) {
      WONDER_plane(previous = bank_sel[row]);
      test = row_start[row];
      }
   if (attr & 0x0100)
      for(i=0; i<text_height; i++, ptr++, row++) {
         for (j=0x80, k=0; j; k++, j >>= 1)
            if (*ptr & j) {
               if (chiptype == TSENG && test > row_start[row]+col+k) {
                  test = row_start[row]+col+k;
                  outp(0x3CD, inp(0x3CD) + 9);
                  }
               if (chiptype == VGAWONDER && test > row_start[row]+col+k) {
                  test = row_start[row]+col+k;
                  WONDER_plane(++previous);
                  }
               pokeb(row_start[row]+col+k, regen, attr);
               }
         }
   else if (attr & 0x0200)
      for(i=0; i<text_height; i++, ptr++, row++) {
         for (j=0x80, k=0; j; k++, j >>= 1)
            if (*ptr & j) {
               if (chiptype == TSENG && ((test ^ row_start[row]+col+k) & 0x8000)) {
                  test = row_start[row]+col+k;
                  outp(0x3CD, inp(0x3CD) + 9);
                  }
               if (chiptype == VGAWONDER && ((test ^ row_start[row]+col+k) & 0x8000)) {
                  test = row_start[row]+col+k;
                  WONDER_plane(++previous);
                  }
               bkcolor = peek(row_start[row]+col+k, regen);
               pokeb(row_start[row]+col+k, regen, attr ^ bkcolor);
               }
         }
   else
      for(i=0; i<text_height; i++, ptr++, row++) {
         for (j=0x80, k=0; j; k++, j >>= 1) {
               if (chiptype == TSENG && ((test ^ row_start[row]+col+k) & 0x8000)) {
                  test = row_start[row]+col+k;
                  outp(0x3CD, inp(0x3CD) + 9);
                  }
               if (chiptype == VGAWONDER && ((test ^ row_start[row]+col+k) & 0x8000)) {
                  test = row_start[row]+col+k;
                  WONDER_plane(++previous);
                  }
            pokeb(row_start[row]+col+k, regen, (*ptr & j)?attr: 0);
            }
         }
   }

eight_get(x, y)
   unsigned int x, y;
   {
   if (chiptype == TSENG)
      outp(0x3CD, bank_sel[y] + (row_start[y]+(long)x > 0x10000L? 9: 0));
   if (chiptype == VGAWONDER)
      WONDER_plane(bank_sel[y] + (row_start[y]+(long)x > 0x10000L? 1: 0));
   return (peek(row_start[y]+x, regen) & 0xFF);
   }

eightcolor(x, y, testcolor)
   unsigned int x, y;
   unsigned int testcolor;
   {
   if (chiptype == TSENG)
      outp(0x3CD, bank_sel[y] + (row_start[y]+(long)x > 0x10000L? 9: 0));
   if (chiptype == VGAWONDER)
      WONDER_plane(bank_sel[y] + (row_start[y]+(long)x > 0x10000L? 1: 0));
   return ((peek(row_start[y] + x, regen) & 0xFF) - testcolor);
   }

void eight_block(image, col,row, sizeX,sizeY)
int col, row;
char huge *image;
int sizeX,sizeY;
/* image is supposed to contain a color for each pixel, one pixel in one byte
   top line has to be stored first
   col,row are coordinates of top left corner
*/
{ char far *dest;
  unsigned addr,left,end,previous;

  FP_SEG(dest) = regen;
  for (end = row+sizeY; row<end; row++, image+=sizeX)
  { if (chiptype == VGAWONDER || chiptype == TSENG)
    { addr = row_start[row];
      left = col + addr;
      if (chiptype == TSENG)
        outp(0x3CD, previous = bank_sel[row] + (addr > left? 9: 0));
      if (chiptype == VGAWONDER)
        WONDER_plane(previous = bank_sel[row] + (addr > left? 1: 0));
      FP_OFF(dest) = left;
      if ((long)left+sizeX > 0x10000L)
      { 
	/* KT 12/01/2000 forget about fmemcpy (old compilers) */
	memcpy(dest, image, -left);  /*left!=0 because sizeX<1024*/
        if (chiptype == TSENG)
          outp(0x3CD, previous += 9);
        if (chiptype == VGAWONDER)
          WONDER_plane(++previous);
        FP_OFF(dest) = 0;
        /* KT 12/01/2000 forget about fmemcpy (old compilers) */
	memcpy(dest,image+0x10000L-left,left+sizeX);
      }
      else
	/* KT 12/01/2000 forget about fmemcpy (old compilers) */
        memcpy(dest,image,sizeX);
    }
    else
    { FP_OFF(dest) = row_start[row]+col;
      /* KT 12/01/2000 forget about fmemcpy (old compilers) */
      memcpy(dest,image,sizeX);
    }
  }
}

eight_bit(col, row)
   unsigned int col, row;
   {
   if (chiptype == TSENG)
      outp(0x3CD, bank_sel[row] + (row_start[row]+(long)col > 0x10000L? 9: 0));
   if (chiptype == VGAWONDER)
      WONDER_plane(bank_sel[row] + (row_start[row]+(long)col > 0x10000L? 1: 0));
   pokeb(row_start[row] + col, regen, color);
   }

newFour_get(x, y)
   unsigned int x, y;
   {
   if (chiptype == VGAWONDER)
      WONDER_plane(bank_sel[y] + (row_start[y]+(long)x/2 > 0x10000L? 1: 0));
   return (four_get(x,y));
   }

newFourcolor(x, y, testcolor)
   unsigned int x, y;
   unsigned int testcolor;
   {
   if (chiptype == VGAWONDER)
      WONDER_plane(bank_sel[y] + (row_start[y]+(long)x/2 > 0x10000L? 1: 0));
   return (fourcolor(x,y,testcolor));
   }

newFour_bit(col, row)
   unsigned int col, row;
   {
   if (chiptype == VGAWONDER)
      WONDER_plane(bank_sel[row] + (row_start[row]+(long)col/2 > 0x10000L? 1: 0));
   four_bit(col,row);
   }

void color_s(row, col, attr, s)
   int row, col, attr;
   char s[];
   {
   int i;
   int c;

   for(i=0; (c = s[i]) != 0; i++) {
      (*color_c)(row, col+i, attr, c);
      }
   }

int set_color(c)
   int c;
   {
   if (zmax <= 16)
      color = color_table[c & 0x0F];
   else
      color = c % zmax;
   return color;
   }
/*************************** start setmode,init**************************/

void setmode(mode)
int mode;
{ mode_t *setup;
  unsigned int i, j;
  unsigned int row;
  unsigned int c, mask1, mask2;
  unsigned int max_scan;

  for (i=0; mode_tbl[i].mode && mode_tbl[i].mode !=mode; i++)
    ;
  if (!mode_tbl[i].mode)
  { printf("mode number not found,aborting");
    exit(1);
  }
  setup = &mode_tbl[i];
  vmode = mode;
  xmax      = setup->pixels;
  ymax      = setup->lines;
  textrows  = setup->rows;
  textcols  = setup->chars;
  regen     = setup->segmt;
  memend    = setup->size;
  zmax = setup->colors;

  text_height = setup->charsize;
  max_scan  = setup->frames;
  bits_per_pixel = setup->bits;

   fontptr = getfont(text_height < 13? 3: text_height < 15? 2: 6);
   row_start[0] = 0x55;
   block = dummy;

   if(bits_per_pixel == 1) {
      if (zmax <= 2) {
         dot = one_bit;             /* draw_dot function */
         get_dot = one_get;         /* read_dot function */
         find_color = onecolor;
         color_table = &one_color[0];  /* color lookup/expand table */
         }
      else {
         dot = ega_bit;             /* draw_dot function */
         get_dot = ega_get;         /* read_dot function */
         find_color = egacolor;
         block = ega_block;
         if (vmode==15)
           color_table = &ega_mono_color[0];
         else
           color_table = &ega_color[0];
         limit1 = 0xFFFF;
         j = xmax/8;
         for(row = i = 0; i < MAX_ROW; ++i) {   /* for each scanline */
            if (row >= 0xE000 && limit1 == 0xFFFF) {
               limit1 = i;
               row -= 0xE000;
               }
            row_start[i] = row;      /* calc the row start address */
            row += j;                /* base addr of group of lines */
            }
         }
      color_c = one_char;           /* text output routine */
      get_shift = &one_shift[0];    /* read_dot alignment shift count table */
      clear_word = 0;               /* buffer clearing value */
      pixels_per_byte = 8;
      }
   else if(bits_per_pixel == 2) {
         pixels_per_byte = 4;
         dot = two_bit;
         color_table = &two_color[0];
         get_dot = two_get;
         find_color = twocolor;
         get_shift = &two_shift[0];
         color_c = two_char;
         clear_word = 0;
         for(c=0; c<256; c++) { /* build the text bit pattern expand table */
            expand_left[c] = 0;
            mask1 = 0x08;
            for(i=0xC000; i; i>>=2) { /* color bit mask */
               if(c & mask1)
                  expand_left[c] |= i;
               mask1 >>= 1;
               if(mask1 == 0)
                  mask1 = 0x80;
               }
            }
         }
      else if (bits_per_pixel == 4) {  /* 4 bits per pixel */
         pixels_per_byte = 2;
         dot = newFour_bit;
         color_table = &four_color[0];
         get_dot = newFour_get;
         find_color = newFourcolor;
         get_shift = &four_shift[0];
         color_c = four_char;
         clear_word = 0;
         for(c=0; c<256; c++) { /* build the text bit pattern expand table */
            expand_left[c] = expand_right[c] = 0;
            mask1 = 0x20;
            mask2 = 0x02;
            for(i=0xF000; i; i>>=4) { /* color bit mask */
               if(c & mask1) expand_left[c] |= i;
               mask1 >>= 1;
               if(mask1 == 0x08) mask1 = 0x80;

               if(c & mask2) expand_right[c] |= i;
               mask2 >>= 1;
               if(mask2 == 0) mask2 = 0x08;
               }
            }
         }
      else {  /* 8 bits per pixel */
         pixels_per_byte = 1;
         dot = eight_bit;
         get_dot = eight_get;
         find_color = eightcolor;
         block = eight_block;
         color_c = eight_char;
         clear_word = 0;
         if (chiptype == VGAWONDER) {
            for(row = i = 0, j=0; i < MAX_ROW; i++) {      /* for each VTOTAL */
               bank_sel[i] = j;            /* calc the row start address */
               if (row > 0xFC00 && (row + xmax) < 0x0400)
                  j ++;
               row += xmax;                /* base addr of group of lines */
               }
            }
         else {
            for(row = i = 0, j=0x40; i < MAX_ROW; i++) {      /* for each VTOTAL */
               bank_sel[i] = j;            /* calc the row start address */
               if (row > 0xFC00 && (row + xmax) < 0x0400)
                  j += 0x09;
               row += xmax;                /* base addr of group of lines */
               }
            }
         }

      row = 0;
      if (row_start[0] == 0x55)
         for(i = 0; i < MAX_ROW; i += max_scan) { /* for each VTOTAL */
            for(j = 0; j < max_scan; j++)      /* for each scan row */
               row_start[i+j] = row + (j<<13); /* calc the row start address */
            row += xmax/pixels_per_byte;       /* base addr of group of lines */
            }

      aspect = ((unsigned)ymax * 80) / xmax;
           /*  ((ymax * 5) * 64) / (xmax * 4)  */
      if(aspect > 63)
         aspect = 63;
}

void init()
{
  vidmode(vmode);
  if (color256())
    /*grey_scale();*/ rainbow();
}
/**************************** end init *******************************/
/*
 *
 * BOX: Draw a (color) rectangle from upper left = (x1,y1)
 *              to lower right = (x2,y2)
 *
 */
void box( x1, y1, x2, y2 )
int x1, y1, x2, y2;
{
  line(x1,y1,x2,y1);
  line(x2,y1,x2,y2);
  line(x2,y2,x1,y2);
  line(x1,y2,x1,y1);
}

void filled_box(x1, y1, x2, y2)
int x1,y1, x2,y2;
{ int t;

  if(y2<y1) {  /* draw box top to bottom */
    t=y2; y2=y1;  y1=t;
    }
  while (y1<=y2) {     /* using horizontal lines       */
    line(x1, y1, x2, y1);
    y1++;
    }
}

shaded_box(x1, y1, x2, y2, color1)
   int x1,y1, x2,y2, color1;
   {
   int t, colorused;
   int change;

   colorused = color1 - 1;
   change = 1;
   if(y2<y1) {  /* draw box top to bottom */
      t=y2; y2=y1; y1=t;
      }
   while (y1<=y2) {     /* using horizontal lines       */
      if ((colorused = (colorused + change) & 0xFF) < color1) {
         change = - change;
         continue;
         }
      set_color(colorused);
      line(x1, y1, x2, y1);
      y1++;
      if (digital_flag) {
         if (y1 <= y2) {
            line(x1, y1, x2, y1);
            y1++;
            }
         if (y1 <= y2) {
            line(x1, y1, x2, y1);
            y1++;
            }
         }
      }
   }

/********************************************************************
 *
 * PAINT: Fill in the convex area containing (x,y) bounded by color (bound)
 *              with color set by last call to set_color().
 *
 *******************************************************************/
#ifndef FILL
paint (xi, yi, bound)
   int xi, yi, bound;
   {
   int yinc;

   if ((*get_dot)(xi, yi) == bound)
      return;
   yinc = -1;
   pass_paint(xi, yi, bound, yinc);
   yinc = 1;
   pass_paint(xi, yi+1, bound, yinc);
   }

pass_paint (xi, yi, bound, yinc)
   int xi, yi, bound, yinc;
   {
   int xl, xk, xn, xp, yp;

   yp = yi;
   xp = xi + 1;
   xl = xi - 1;
   if ((*find_color)(xi, yp, bound) == 0) {
      do {
         xl -= pixels_per_byte;
         if (xl < 0) {
            xl = 0;
            break;
            }
         } while ((*find_color)(xl, yp, bound) == 0);
      xl += pixels_per_byte;
      do {
         xp += pixels_per_byte;
         if (xp > xmax) {
            xp = xmax;
            break;
            }
         } while ((*find_color)(xp, yp, bound) == 0);
      xp -= pixels_per_byte;
      }
   while ((*get_dot)(xl, yp) != bound)
      if (--xl < 0)
         break;
   xl++;
   while ((*get_dot)(xp, yp) != bound)
      if (++xp > xmax)
         break;
   xp--;

   while (1) {
      line(xl, yp, xp, yp);
      yp += yinc;
      if (yp >= ymax || yp < 0)
         return;
#ifdef NONCONVEX
      while ((*get_dot)(xl, yp) == bound) {
         /* loss on the left */
         if (++xl > xp)
            return;
         }
      while ((*get_dot)(xl - 1, yp) != bound) {
         /* growth to left */
         if (--xl == 0)
            break;
         }
      while ((*get_dot)(xp, yp) == bound) {
         /* loss on the right */
         --xp;
         }
      while ((*get_dot)(xp + 1, yp) != bound) {
         /* growth to right */
         if (++xp == xmax)
            break;
         }
      pass_loop:
      xk = (xl + pixels_per_byte - 1)&(- pixels_per_byte);
      if (xk >= xp)
         xk = xp;
      for (xn = xl; xn < xk; xn++) {
         if ((*get_dot)(xn, yp) == bound) {
            printf("Recursive Paint Call (%d-%d, %d)\n", xl, xn-1, yp);
            pass_paint(xl, yp, bound, yinc);
            while ((*get_dot)(xn, yp) == bound)
               xn++;
            xl = xn;
            }
         }
      for (xn = xk; xn < xp; xn += pixels_per_byte) {
         if ((*find_color)(xn, yp, bound)) {
            while ((*get_dot)(xn, yp) != bound)
               xn++;
            xn--;
            if (xn < xk) {
               printf("Recursive Paint Call (%d-%d, %d)\n", xl, xn-1, yp);
               pass_paint(xl, yp, bound, yinc);
               while ((*get_dot)(xn, yp) == bound)
                  xn++;
               xl = xn;
               goto pass_loop;
               }
            break;
            }
         }
#else
      if ((*get_dot)(xl, yp) != bound) {
         do {
            xl--;
            } while(xl >= 0 && (*get_dot)(xl, yp) != bound);
         xl++;
         }
      else
         do {
            xl++;
            } while(xl < xp && (*get_dot)(xl, yp) == bound);
      if ((*get_dot)(xp, yp) == bound)
         do {
            xp--;
            } while(xp >= xl && (*get_dot)(xp, yp) == bound);
      else {
         do {
            xp++;
            } while(xp < xmax && (*get_dot)(xp, yp) != bound);
         xp--;
         }
      if (xl > xp)
         break;
#endif
      }
   }
#else
paint (xi, yi, bound)
   int xi, yi, bound;
   {
   fill(xi, yi, color);
   }
#endif
void ellipse(x, y, major, aspect)
    int x, y, major;
    unsigned int aspect;          /* aspect ratio * 256 */
    {
    int row, col, two_x, two_y;
    long d, cincr, rincr, two_a, two_b, alpha, beta;

    d = ((long)major)*((long)aspect);
/* the following code applies if "major" is the major axis length */
#ifdef REALMAJOR
    if (aspect < 256) {
        two_a = ((long)major)*((long)major)*2L;
        two_b = ((d >> 3) * (d >> 3)) >> 9;
        d = (d+64) >> 7;
        }
    else {
        two_a = ((d >> 4) * (d >> 4)) >> 7;
        two_b = ((long)major)*((long)major)*2L;
        d = major * 2;
        }
#else
/* the following code applies if "major" is the x-axis length */
    two_a = ((long)major)*((long)major)*2L;
    two_b = ((d >> 4) * (d >> 4)) >> 7;
    d = (d+64) >> 7;
#endif
    row = y + (d >> 1);
    col = x;
    two_x = x << 1;
    two_y = y << 1;
    alpha = two_a >> 1;
    beta = two_b >> 1;
    rincr = - (alpha * d);
    cincr = beta;
    d = ((beta - alpha) << 1) + (alpha >> 1);
/*
 *  For the remaining pare of the routine, only 'col', 'row', 'two_y',
 *  'two_x', 'color', 'two_a', 'two_b', 'cincr', 'rincr' and 'd' are
 *  use.
 */
    while (cincr + rincr < 0) {
        (*dot)(col, row);
        (*dot)(col, two_y - row);
        (*dot)(two_x - col, row);
        (*dot)(two_x - col, two_y - row);
        d += (cincr += two_b);
        col++;
        if (d >= 0) {
            d += (rincr += two_a);
            row--;
            }
        }
    while (row >= y) {
        (*dot)(col, row);
        (*dot)(col, two_y - row);
        (*dot)(two_x - col, row);
        (*dot)(two_x - col, two_y - row);
        if (d <= 0) {
            d += (cincr += two_b);
            col++;
            }
        row--;
        d += (rincr += two_a);
        }
    }

void line(x1, y1, x2, y2)
   int x1, y1, x2, y2;
   {
   int i1, i2, d1, d2, d;
   unsigned addr,previous;
   unsigned int left, right, x3, x4;

   if (dot == ega_bit)
      (*dot)(x1, y1);
   d2 = y2 - y1;
   d1 = x2 - x1;
   if(d2 == 0) {    /* horizontal line */
      if(d1 < 0) {  /* set drawing direction */
         left = x2;
         right = x1;
         }
      else {
         left = x1;
         right = x2;
         }
      if (bits_per_pixel == 8) {
         addr = row_start[y1];
         left += addr;
         right += addr;
         color |= color << 8;
         if (chiptype == TSENG)
            outp(0x3CD, previous = bank_sel[y1] + (addr > left? 9: 0));
         if (chiptype == VGAWONDER)
            WONDER_plane(previous = bank_sel[y1] + (addr > left? 1: 0));
         if ((unsigned)left > (unsigned)right) {
            filler(left, regen, 0xFFFF, color);
            if (chiptype == TSENG)
              outp(0x3CD, previous += 9);
            if (chiptype == VGAWONDER)
              WONDER_plane(++previous);
            filler(0, regen, right, color);
            }
         else {
            filler(left, regen, right, color);
            }
         return;
         }
      if(left+16 >= right) {   /* quickest way to draw short lines */
         for(x1 = left; x1<= right; x1++)
            (*dot)(x1, y1);
         return;
         }

      x1 = left;
      x4 = right;
      if(bits_per_pixel == 1) {
         x2 = (left  + 7) & 0xFFF8;
         x3 = (right - 7) & 0xFFF8;
         left = x2 / 8;
         right = x3 / 8 - 1;
         }
      else if(bits_per_pixel == 2) {
         x2 = (left  + 3) & 0xFFFC;
         x3 = (right - 3) & 0xFFFC;
         left = x2 / 4;
         right = x3 / 4 - 1;
         }
      else if(bits_per_pixel == 4) {
         x2 = (left  + 1) & 0xFFFE;
         x3 = (right - 1) & 0xFFFE;
         left = x2 / 2;
         right = x3 / 2 - 1;
         }

      addr = row_start[y1];
      if(left <= right) {
         if (dot == ega_bit) {
            wrt_ega(0x0205,0x3CE);
            wrt_ega(0xFF08,0x3CE);
            }
         if (bits_per_pixel==4) {
            left += addr;
            right += addr;
            color |= color<<8;
            if (chiptype == VGAWONDER)
               WONDER_plane(previous = bank_sel[y1] + (addr > left? 1: 0));
            if ((unsigned)left > (unsigned)right) {
               filler(left, regen, 0xFFFF, color);
               if (chiptype == VGAWONDER)
                  WONDER_plane(++previous);
               filler(0, regen, right, color);
               }
            else
               filler(left, regen, right, color);
            }
         else
            filler(addr + left, regen, addr + right,
                  color | (color << 8)); /* middle of line */
         }
      if(x1 < x2)
         for( ; x1 < x2; x1++)
            (*dot)(x1, y1); /* left edge */
      if(x3 <= x4)
         for( ; x3 <= x4; x3++)
            (*dot)(x3, y1); /* right edge */
      return;
      }

   i1 = 1;
   if (d1 < 0) {
      d1 = -d1;
      i1 = -1;
      }
   i2 = 1;
   if (d2 < 0) {
      d2 = -d2;
      i2 = -1;
      }

   if(d1 > d2) {
      d = d2 + d2 - d1;
      while(1) {
         (*dot)(x1, y1);
         if(x1 == x2)
            break;
         if(d >= 0) {
            d -= (d1 + d1);
            y1 += i2;
            }
         d += d2 + d2;
         x1 += i1;
         }
      }
   else {
      d = d1 + d1 - d2;
      while (1) {
         (*dot)(x1, y1);
         if(y1 == y2)
            break;
         if(d >= 0) {
            d -= (d2 + d2);
            x1 += i1;
         }
         d += d1 + d1;
         y1 += i2;
      }
   }
}
#endif /* SC_PC */

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
