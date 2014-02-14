/* 
*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2001, IRSL
    See STIR/LICENSE.txt for details
*/
/*!
 \file 
  
 \brief  Declares some utility functions used by the screen* files
  
 \author Kris Thielemans
 \author PARAPET project
 
  


 This is part of a library by Kris Thielemans, mainly written in 1991.
  
 \internal

   Standard include file where some incompatibilities between various
   systems are taken care of (together with gen.c).

   Currently this supports:
     VAX VMS
     	#ifdef VAX
     ultrix  on MIPS 
     Solaris on Sparc
     OSF/1 on Alpha
     AIX on PowerPC
       #ifdef ultrix
       (for historical reasons, all unix machines)
     MSDOS (checking MSC, or gcc)
       #ifdef MSDOS
     PowerMAC (using Metroworks Codewarrior Pro) [no keyboard and
graphics yet]
       this last one is a cross-compiler for PPC, 68K and Win NT on x86
systems.
       Only checked it on PPC though.
       #ifdef __MSL__  (Metroworks Standard Library)

   As a convenience we include stdio.h and stdlib.h  here.


*/
#include <stdio.h>

/* Change November 1997: added next 3 lines */
#ifdef __cplusplus
extern "C" {
#endif

#ifdef VAX
/* The VAX library does not have getch. An implemenation is provided in gen.c
  getch reads a character from the keyboard without waiting for a
        carriage return. You can use it for detecting arrowkeys
        in combination with the KB_... macro's.
*/
extern  char getch(void);
/* Macro's for using the arrowkeys. Use in the following way:
        char ch;
        ch = getch();
        if (KB_DIRECTION(ch))
        { // ch is now the last character of the escape code
          switch(ch)
          { KB_UPARROW: ...
            ...
          }
        }
        else ...
         // ch is not changed, or the second character in the escape code
  Note: The following are ANSI escape sequences
*/
#define KB_DIRECTION(c) (c==0x1b && getch()==0x5b && \
        (c=getch())>=0x41 && c<=0x44)
#define KB_UPARROW 0x41
#define KB_DNARROW 0x42
#define KB_RTARROW 0x43
#define KB_LTARROW 0x44
#endif /* VAX */

#if (defined(_WIN32) || defined(WIN32)) && !defined(MSDOS)
#define MSDOS
#endif

#ifdef MSDOS
  /* Change 05/02/98 */
#ifdef __GNUC__
  /* TODO */
#define getch() getchar()
#else
#include <conio.h>              /* for getch */
#define getch _getch
#endif

/* Macro's for using the arrowkeys. For explanation see above.
  Replace "escape code" with "extended charcode"
*/
#define KB_DIRECTION(c) (c=='\0' && \
                         ((c=(char)getch())==0x48 || c==0x4b ||c==0x4d || \
			  c==0x50))
#define KB_UPARROW 0x48
#define KB_DNARROW 0x50
#define KB_RTARROW 0x4d
#define KB_LTARROW 0x4b
#endif /* MSDOS */

#if !defined(VAX) && !defined(MSDOS) && !defined(__MSL__)
#include <curses.h>             /* for getch */
  /* Change November 1997: added 7 lines
     work-around a 'bug'. screen.h includes curses.h which #defines some
     symbols which conflict in other C++ includes
  */

#ifdef clear
#undef clear
#endif
#ifdef erase
#undef erase
#endif

/* Macro's for using the arrowkeys. For explanation see above.
*/
#define KB_DIRECTION(c) (c==0x1b && getch()==0x5b && \
        (c=getch())>=0x41 && c<=0x44)
#define KB_UPARROW 0x41
#define KB_DNARROW 0x42
#define KB_RTARROW 0x43
#define KB_LTARROW 0x44
#endif /*ultrix*/

#ifdef SC_XWINDOWS
#undef KB_DIRECTION
#undef KB_UPARROW
#undef KB_DNARROW
#undef KB_RTARROW
#undef KB_LTARROW
/* the next one is probably wrong */
#define KB_DIRECTION(c) (c==0x1b && getch()==0x5b && \
        (c=getch())>=0x41 && c<=0x44)
#define KB_UPARROW XK_Up
#define KB_DNARROW XK_Down
#define KB_RTARROW XK_Right
#define KB_LTARROW XK_Left
#endif

#include <stdlib.h>

#ifndef Min
#define Min(x,y) ((x)<(y) ? (x) : (y))
#endif
#ifndef Max
#define Max(x,y) ((x)>(y) ? (x) : (y))
#endif

/**************************************************************************

***************************************************************************/
extern size_t fread_check       (char str[],void * buffer,size_t size,FILE *infile);
extern size_t fwrite_check      (char str[],void * buffer,size_t size,FILE *outfile);
extern void   fseek_check       (char str[],FILE *file, long offset, int pos);

extern int asknr (char str[],int minv,int maxv,int def);

extern void message(char *fmt, ...);
extern void error  (char *fmt, ...);


/* Change November 1997: added next 3 lines (end of extern "C") */
#ifdef __cplusplus
}
#endif
