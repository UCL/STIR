/* Standard include file where all incompatibilities between various
   systems are taken care of (together with gen.c).

   Currently this supports:
     VAX VMS (ANSI C)
     	#ifdef VAX
     ultrix  on MIPS (this was at the time halfway ANSI C)
     Solaris on Sparc (ANSI C)
     OSF/1 on Alpha (ANSI C)
     AIX on PowerPC
       #ifdef ultrix
       (for historical reasons, all unix machines)
     MSDOS (assuming MSC -> ANSI C)
       #ifdef MSDOS
     PowerMAC (using Metroworks Codewarrior Pro-> ANSI C) [no keyboard and
graphics yet]
       this last one is a cross-compiler for PPC, 68K and Win NT on x86
systems.
       Only checked it on PPC though.
       #ifdef __MSL__  (Metroworks Standard Library)

   As a convenience we include stdio.h and stdlib.h (for ANSI C) here.
*/
#include <stdio.h>

/* Change November 1997: added next 3 lines */
#ifdef __cplusplus
extern "C" {
#endif

/* Use macros for standard return values of 'main */
#ifdef VAX
#ifndef SS$_NORMAL
#include ssdef
#endif
#define EXIT_OK SS$_NORMAL
#define EXIT_ABORT SS$_ABORT
#else
#define EXIT_OK 0
#define EXIT_ABORT 1
#endif

#ifdef VAX
#define ANSI
#define INT32 long
#define PACKED_STRUCTS
extern void mysort(void *base,size_t nmemb,size_t size,
                   int (*comp)(void *, void *));
#define qsort mysort
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

#ifdef MSDOS
#define INT32 long
#define ANSI
#define PACKED_STRUCTS
#include <conio.h>              /* for getch */
#include <search.h>             /* for qsort */
/* Macro's for using the arrowkeys. For explanation see above.
  Replace "escape code" with "extended charcode"
*/
#define KB_DIRECTION(c) (c=='\0' && \
                         ((c=(char)getch())==0x48 || c==0x4b ||c==0x4d ||
c==0x50))
#define KB_UPARROW 0x48
#define KB_DNARROW 0x50
#define KB_RTARROW 0x4d
#define KB_LTARROW 0x4b
#endif /* MSDOS */

#if defined(__STDC__) && !defined(ANSI)
#define ANSI
#endif

#ifdef __MSL__
#if macintosh && !defined(__dest_os)
  #define __dest_os __mac_os
#endif
#define INT32 long
#endif

#ifdef ultrix
#define INT32 long
#ifndef ANSI
void qsort();
#endif
#endif

#if defined(__osf__) || defined(__unix__) || defined(sun)
#define ultrix
#define INT32 int
#endif

#ifdef ultrix
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

#ifndef ANSI
#define const
#define size_t unsigned
#define VOIDP char *
char *strlwr(char *);           /* convert to lowercase */
char *strupr(char *);           /* convert to uppercase */
#else
#define VOIDP void *
#include <stdlib.h>
#endif

#ifndef Min
#define Min(x,y) ((x)<(y) ? (x) : (y))
#endif
#ifndef Max
#define Max(x,y) ((x)>(y) ? (x) : (y))
#endif

/*************************************************************************
  Routines for swapping between littleendian and bigendian machines
  Added April 97

  swaporderf is the same as swaporder4, but for floats. It uses a
  union to avoid compiler errors and strange sign extensions.
 ************************************************************************/
#define swaporder2(a) (a = ((0xff00 & a) >> 8) | ((0xff & a) << 8))
#define swaporder4(a) (a = \
          ((0xff000000 & (INT32)a) >> 24) | ((0xff0000 & (INT32)a) >> 8) |\
          ((0xff00 & (INT32)a) << 8) | ((0xff & (INT32)a) <<24))
#define swaporderf(a) { union {float f; INT32 i;} swaptemp; \
          swaptemp.f = a; swaporder4(swaptemp.i); a = swaptemp.f; }


/**************************************************************************
   routines for conversion from VAX floating point format to own format
   (note: not for "double" !)
   usage: float a; VAXfloat Va;
        a = (float)VAXfl_to_fl(Va);     // type conversion to avoid warnings
        Va = fl_to_VAXfl(a);
        makefloat(a);
        // a is a float that contains the number in VAX format
        makeVAXfloat(a);
        // a is a float, content is converted to VAX format
***************************************************************************/
#ifndef VAX
/* This gets complicated by different architectures: little/big endian
   machines. On Solaris (SVR4?) you could check this with a
   #ifdef _BIGENDIAN,
   on Silicon Graphics (Berkeley Unix ?) you would do
   #if BYTEORDER == BIGENDIAN
   on Metroworks CodeWarrior
   #ifndef __LITTLE_ENDIAN
   Oh well.

   One would expect that a swaporderf would make any further distinction
   unnecessary. This is not true as bit fields are assigned in a
   different order on different architectures (in principle this is
   a compiler choice and not necessarily related to little/big endian,
   but in practice, this two features are related on the PCs, Decs, Alphas
   and Suns.
*/

#if defined(sparc) || defined(__sparc) || \
   (defined(__MSL__) && !defined(__LITTLE_ENDIAN))
/* TODO add PowerPC AIX */
/* definition for bigendian machines.
   Do a swaporderf(a) first before using this bit field.
*/
typedef struct
        {
          unsigned frc2 : 16;
          unsigned sign : 1;
          unsigned exp  : 8;
          unsigned frc1 : 7;
        } VAXfloat;
#else
/* definition for littleendian machines.
   if the float is stored in memory as b1 b2 b3 b4, then
   frc2 = b4 b3
   sign = first bit of b2
   frc1 = last 7 bits of b1
   exp = (last 7 bits of b2) << 2 | (first bit of b1)

   in bit-field language this is all turned around...
   */
typedef struct
        { unsigned frc1 : 7;
          unsigned exp  : 8;
          unsigned sign : 1;
          unsigned frc2 : 16;
        } VAXfloat;
#endif

extern double VAXfl_to_fl(VAXfloat Va);
extern VAXfloat fl_to_VAXfl(double a);
#define makefloat(a)    a = (float)VAXfl_to_fl(*((VAXfloat *)&a))
#define makeVAXfloat(a) *((VAXfloat *)&a) = fl_to_VAXfl(a)

#else /*VAX */

/* on VAX, float conversion is not necessary, so do nothing */
typedef float VAXfloat;
#define VAXfl_to_fl(Va) (Va)
#define fl_to_VAXfl(a)  (a)
#define makefloat(a)
#define makeVAXfloat(a)
#endif /* VAX */


/************************************************************************
  The following are routines defined in the MSC run-time library.
  We define them here for other machines.
  _splitpath expects a full pathname, and returns the different components
        of it. If there is no drive,dir ... specified, the corresponding
        string will be a null string.
  _makepath assembles the different components into a pathname.
        If you don't want to specify a drive,dir ..., you can use null
        strings or NULL pointers.
  The _MAX_... constants are now given as reasonable values, not real
        restrictions. If you need them higher, Try it !
 ************************************************************************/
#ifndef MSDOS
#define _MAX_DRIVE 20
#define _MAX_DIR   90
#define _MAX_FNAME 20
#define _MAX_EXT   20
extern  void _splitpath
        (char *path,char *drive,char *dir,char *fname,char *ext);
extern  void _makepath
        (char *path,char *drive,char *dir,char *fname,char *ext);
#endif /* MSDOS */

#ifdef ultrix
#define WRITEBIN "w"
#define APP_WRITEBIN "a"
#define READBIN "r"
#else
#define WRITEBIN "wb"
#define APP_WRITEBIN "ab+"
#define READBIN "rb"
#endif
#define OPENWBIN(file) fopen(file,WRITEBIN)
#define OPENRBIN(file) fopen(file,READBIN)

/**************************************************************************
  returns number of days since 0-1-1900,
  date format: 10-JUN-89 (can be lower case)
***************************************************************************/
extern long days_since_1900     (char date[]);
/**************************************************************************

***************************************************************************/
extern long time_to_secs        (char str[]);
extern char *secs_to_time       (char str[], long secs);
/**************************************************************************

***************************************************************************/
extern VOIDP  malloc_check      (char str[],size_t size);
extern VOIDP  realloc_check     (char str[],VOIDP buffer, size_t size);
extern size_t fread_check       (char str[],VOIDP buffer,size_t size,FILE
*infile);
extern size_t fwrite_check      (char str[],VOIDP buffer,size_t size,FILE
*outfile);
extern void   fseek_check       (char str[],FILE *file, long offset, int pos);
#ifdef VAX
extern FILE * fopen_check       (char str[],char *filename, char *mode, ...);
#else
extern FILE * fopen_check       (char str[],char *filename, char *mode);
#endif
extern long   filesize          (FILE *f);
/**************************************************************************

***************************************************************************/
extern void skipCR              (void);
/**************************************************************************

***************************************************************************/
extern int  ask                 (char prompt[], char def);
#ifdef ANSI
extern char askchoice           (char prompt[], char def, char all[], ...);
#else
extern char askchoice           ();
#endif
extern long asklong             (char prompt[], long minl, long maxl, long def);
extern int  asknr               (char prompt[], int mini, int maxi, int def);
extern double askdouble         (char prompt[],
                                 double minimum,double maximum,double def);
extern char *askstr             (char prompt[], char *str, int length);
/**************************************************************************

***************************************************************************/
extern char *find_filename      (char *str);
/**************************************************************************

***************************************************************************/
extern char *add_extension      (char *str,char *def);
extern char *asknm_extension    (char *str,char *name,char *def);
extern FILE *askfile_extension  (char *str,char *def,char *mode);
/* old routines for ECAT II and IV files.
   They don't really belong here, sorry. */
extern char *get_pat_file       (char *str,char *dir,int pat);
extern char *get_scan_name      (char *str,char *dir,int pat,int scan);
extern char *get_scan_file      (char *str,char *dir,int pat,int scan);
extern  void get_pat_and_scan   (char *fname,int *pat,int *scan);
extern char *get_scandat_file   (char *str,char *dir,int pat,int scan);
/**************************************************************************

***************************************************************************/
/* This copies 'dir' into 'str' and appends the relevant directory separator
   if necessary.
*/
extern char *cons_dir           (char *str,char *dir);
/**************************************************************************

***************************************************************************/
/* Returns CPU time used by the current process in units of 10ms (not to
   that accuracy though).
   MacOS and MSDOS don't know about CPU_time, so wall clock time is returned.
   */
   extern unsigned long CPU_time   (void);
/**************************************************************************

***************************************************************************/
#ifdef ANSI
void message(char *fmt, ...);
void error  (char *fmt, ...);
#else
void message();
void error  ();
#endif
#ifdef MSDOS
extern void far *fmemcpy(void far *outptr,void far *buf,int nr);
#endif



/* Change November 1997: added next 3 lines (end of extern "C") */
#ifdef __cplusplus
	   }
#endif
