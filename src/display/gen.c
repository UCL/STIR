#include "gen.h"
#include <string.h>
#include <ctype.h>
#include <time.h>
#ifdef VAX
#include ssdef
#include descrip
#include iodef
#include dcdef
#else
/* Modena library defines malloc in stdlib.h, others not ? */
#ifndef __MSL__
#include <malloc.h>
#endif
#endif
#ifdef ANSI
#include <stdarg.h>
#else
#include <varargs.h>
#endif

#ifdef VAX
static $DESCRIPTOR(TTdevice,"TT");
static int TTchannel;

char getch()
{ char chr;
  unsigned int iosb[2];

  if (TTchannel==0)
    error_check(sys$assign(&TTdevice,&TTchannel,0,0));
  error_check(sys$qiow(0,TTchannel,IO$_READVBLK|IO$M_ESCAPE|IO$M_NOECHO,
                &iosb,0,0,&chr,sizeof(chr),0,0,0,0));
  return (chr);
}

#define STRNCPY(dest,source,n) strncpy(dest,source,n); dest[n]='\0';

void _splitpath(path, drive,dir,fname,ext)
char path[],drive[],dir[],fname[],ext[];
{ char *c,*orpath;

  orpath = path;
  c = strchr(path,':');
  if (c==NULL)
    *drive = '\0';
  else
  { c++;
    STRNCPY(drive,path,c-path);
    path = c;
  }
  if (*path == '[')
  {  c = strchr(path,']');
    if (c==NULL)
      error("_splitpath: invalid path: %s",orpath);
    else
    { c++;
      STRNCPY(dir,path,c-path);
      path = c;
    }
  }
  else
    *dir = '\0';

  c = strchr(path,'.');
  if (c==NULL)
  { *ext = '\0';
    strcpy(fname,path);
  }
  else
  { STRNCPY(fname,path,c-path);
    strcpy(ext,c);
  }
}
#undef STRNCPY

void _makepath(path, drive,dir,fname,ext)
char path[],drive[],dir[],fname[],ext[];
{
  if (drive != NULL)
  { strcpy(path,drive);
    if (strchr(path,':')==NULL)
      strcat(path,":");
  }
  else
    path[0] = '\0';
  if (dir != NULL)
    strcat(path,dir);
  strcat(path,fname);
  if (ext != NULL && *ext != '\0')
  { if (*ext != '.' && strchr(fname,'.')==NULL)
      strcat(path,".");
    strcat(path,ext);
  }
}

/* Substitute for library routine "qsort", VAX C 3.0 seems to have
   a bug there. Problem occurs if it has to sort items where several
   are equal. If in new version OK, you can delete the
        #define qsort mysort
   line in gen.h
   Note: this is no quick sort
*/
void mysort(base,nmemb,size,comp)
VOIDP base;
size_t  nmemb,size;
int     (*comp)(VOIDP ,VOIDP );
{
  char  *dummy,*max,*first,*second;

  dummy = malloc(size);

  for (first = (char *)base + (nmemb-1)*size; first>base; first -= size)
  { for (second = first-size, max = first; second>=base; second -= size)
      if  (comp(max,second) < 0)
        max = second;
    if (max!=first)
    { memcpy(dummy,first,size);
      memcpy(first,max,size);
      memcpy(max,dummy,size);
    }
  }
  free(dummy);
}
#endif /* VAX */

#ifdef ultrix

#define STRNCPY(dest,source,n) strncpy(dest,source,n); dest[n]='\0';

void _splitpath(path, drive,dir,fname,ext)
char path[],drive[],dir[],fname[],ext[];
{ char *c,*orpath;

  orpath = path;
  *drive = '\0';        /* drive has no meaning in UNIX computers */
  c=strrchr(path,'/');
  if (c==NULL)
    /* KT 9 Nov 97, corrected bug. original line
      dir = '\0';
    */
    *dir = '\0';
  else
  { c++;
    STRNCPY(dir,path,c-path);
    path = c;
  }

  c = strchr(path,'.');
  if (c==NULL)
  { *ext = '\0';
    strcpy(fname,path);
  }
  else
  { STRNCPY(fname,path,c-path);
    strcpy(ext,c);
  }
}
#undef STRNCPY

void _makepath(path, drive,dir,fname,ext)
char path[],drive[],dir[],fname[],ext[];
{
  path[0] = '\0';
  if (dir != NULL)
    strcat(path,dir);
  strcat(path,fname);
  if (ext != NULL && *ext != '\0')
  { if (*ext != '.' && strchr(fname,'.')==NULL)
      strcat(path,".");
    strcat(path,ext);
  }
}
#endif /*ultrix*/

#ifdef __MSL__

#define STRNCPY(dest,source,n) strncpy(dest,source,n); dest[n]='\0';

void _splitpath(path, drive,dir,fname,ext)
char path[],drive[],dir[],fname[],ext[];
{ char *c,*orpath;

  orpath = path;
#if __dest_os	== __win32_os
  c = strchr(path,':');
  if (c==NULL)
    *drive = '\0';
  else
  { c++;
    STRNCPY(drive,path,c-path);
    path = c;
  }
  c=strrchr(path,'\\');
#else
  *drive = '\0';        /* drive has no meaning in MacOS */
  c=strrchr(path,':');
#endif
  if (c==NULL)
    *dir = '\0';
  else
  { c++;
    STRNCPY(dir,path,c-path);
    path = c;
  }

  c = strchr(path,'.');
  if (c==NULL)
  { *ext = '\0';
    strcpy(fname,path);
  }
  else
  { STRNCPY(fname,path,c-path);
    strcpy(ext,c);
  }
}
#undef STRNCPY

void _makepath(path, drive,dir,fname,ext)
char path[],drive[],dir[],fname[],ext[];
{
  if (drive != NULL)
  { strcpy(path,drive);
    if (strchr(path,':')==NULL)
      strcat(path,":");
  }
  else
    path[0] = '\0';

  if (dir != NULL)
    strcat(path,dir);
  strcat(path,fname);
  if (ext != NULL && *ext != '\0')
  { if (*ext != '.' && strchr(fname,'.')==NULL)
      strcat(path,".");
    strcat(path,ext);
  }
}
#endif /*__MSL__*/

#ifndef VAX
/* routines for converting VAX floating point format into own format.
   I've chosen for a generic form that should work on all machines
   (it also works on VAX), instead of a quick, but dependent one.
   If you need speed, ... do it yourself.
*/
#include <math.h>
double VAXfl_to_fl(Va)
VAXfloat Va;
{ int sign;

  if (Va.sign)
    sign = -1;
  else
  { if (Va.exp==0)
      return(0.0);
    sign = 1;
  }
  return (sign*ldexp((double)((0x800000 | Va.frc2) + (Va.frc1 * 0x10000)),
                      Va.exp-128-24));
}
VAXfloat fl_to_VAXfl(a)
double a;
{ unsigned long imant;
  double dmant;
  int exp;
  VAXfloat Va;

  if (!a)
  { Va.sign = Va.exp = 0;
    return (Va);
  }
  if (a>0.0)
    Va.sign = 0;
  else
  { Va.sign = 1;
    a = -a;
  }
  dmant = frexp(a,&exp);
  if (exp<-127)
  { Va.sign = Va.exp = 0;
    return (Va);
  }
  if (exp>127)
    error("Floating point number %g too big for VAX format",a);

  Va.exp = exp + 128;
  imant = (unsigned long)0x7fffff & (unsigned long)ldexp(dmant,24);
  /* Compiler can give "data conversion" warnings on the following,
     but it's OK */
  Va.frc1 = (unsigned)(imant >>16);
  Va.frc2 = (unsigned)(imant & 0xffff);
  return(Va);
}

#endif /* no VAX */

#ifndef ANSI
char *strlwr(str)
char *str;
{ char *c;

  for (c=str; *c; c++)
    *c = tolower(*c);
  return str;
}

char *strupr(str)
char *str;
{ char *c;

  for (c=str; *c; c++)
    *c = toupper(*c);
  return str;
}
#endif

VOIDP  malloc_check(str,size)
char str[];
size_t size;
{ char *P;

  if ((P=(char *)malloc(size))==NULL)
    error("%s: error allocating %d bytes",str,size);
  return ((VOIDP )P);
}

VOIDP  realloc_check(str, buffer, size)
char *str;
VOIDP buffer;
size_t size;
{ char *new_buffer;

  if ((new_buffer = realloc(buffer,size)) == NULL)
    error("%s: No space for reallocating %d bytes",
                str,size);
  return(new_buffer);
}


size_t fread_check(str,buffer,size,infile)
char str[];
VOIDP buffer;
size_t size;
FILE *infile;
{ size_t ret;
  char tmp[200];

  ret = fread(buffer,sizeof(char),size,infile);
  if (ferror(infile))
  { sprintf(tmp,"\n%s: error reading for %u bytes, %u read\nreason",
                str,size,ret);
    perror(tmp);
    error("");
  }
  return(ret);
}

size_t fwrite_check(str,buffer,size,outfile)
char str[];
VOIDP buffer;
size_t size;
FILE *outfile;
{ size_t ret;
  char tmp[200];

  ret = fwrite(buffer,1,size,outfile);
  if (ret < size || ferror(outfile))
  { sprintf(tmp,"\n%s: error writing for %u bytes, %u written\nreason",
                str,size,ret);
    if (ferror(outfile))
      perror(tmp);
    error("");
  }
  return(ret);
}

void fseek_check(str,file, offset,pos)
char *str;
FILE *file;
long offset;
int pos;
{ if (fseek(file,offset,pos))
    error("%s: error seeking in file to position %lu",str,offset);
}

#ifdef VAX
FILE *fopen_check(str,filename, mode, ...)
char *str,*filename,*mode;
{ FILE *f;
  char tmp[200];
  va_list ptr;
  int count,i;
#define __fopen_MAX 7
#define va_count(count)         vaxc$va_count (&count)
  char *ptrs[__fopen_MAX];

  va_start(ptr,mode);
  va_count(count);
  count -= 3;
  if (count > __fopen_MAX)
  { message(
    "fopen_check: warning: %s, only used the first %d additional parameters",
    str, __fopen_MAX);
    count = __fopen_MAX;
  }
  for (i=0; i<count; i++)
    ptrs[i] = va_arg(ptr,char *);

  switch(count)
  { case 0:
        f=fopen(filename,mode); break;
    case 1:
        f=fopen(filename,mode,ptrs[0]); break;
    case 2:
        f=fopen(filename,mode,ptrs[0],ptrs[1]); break;
    case 3:
        f=fopen(filename,mode,ptrs[0],ptrs[1],ptrs[2]); break;
    case 4:
        f=fopen(filename,mode,ptrs[0],ptrs[1],ptrs[2],ptrs[3]); break;
    case 5:
        f=fopen(filename,mode,ptrs[0],ptrs[1],ptrs[2],ptrs[3],ptrs[4]); break;
    case 6:
        f=fopen(filename,mode,ptrs[0],ptrs[1],ptrs[2],ptrs[3],ptrs[4],ptrs[5]);
	break;
  }
#undef __fopen_MAX

  if (f==NULL)
  { sprintf(tmp,"\n%s: error opening %s, mode %s\nreason",
                str,filename,mode);
    perror(tmp);
    error("");
  }
  return(f);
}
#else

FILE *fopen_check(str,filename, mode)
char *str,*filename,*mode;
{ FILE *f;
  char tmp[200];

  f=fopen(filename,mode);

  if (f==NULL)
  { sprintf(tmp,"\n%s: error opening %s, mode %s\nreason",
                str,filename,mode);
    perror(tmp);
    error("");
  }
  return(f);
}
#endif

long filesize(file)
FILE *file;
{ char *proc = "filesize";

  fseek_check(proc,file, 0L, 2);
  return ftell(file);
}

static struct MONTH { char m[4]; int days;} MONTHS[12] =
  { {"JAN", 31},    {"FEB", 28},
    {"MAR", 31},    {"APR", 30},
    {"MAY", 31},    {"JUN", 30},
    {"JUL", 31},    {"AUG", 31},
    {"SEP", 30},    {"OCT", 31},
    {"NOV", 30},    {"DEC", 31}};

long time_to_secs(str)
/* Converts hh:mm:ss to seconds after midnight                          */
char str[];
{ int hr,min,sec;

  hr = min = sec = 0;
  sscanf(str, " %d:%d:%d",&hr,&min,&sec);
  return ( (hr*60+min)*60L + sec);
}
char *secs_to_time(str,secs)
/* Converts seconds to hh:mm:ss                                         */
char str[];
long secs;
{ int hr,min;

  hr = (int)(secs / 3600);
  secs %= 3600;
  min = (int)(secs / 60);
  secs %= 60;
  sprintf(str, " %d:%02d:%02d",hr,min,(int)secs);
  return (str);
}

long days_since_1900(date)
/* returns number of days since 0-1-1900,
   date format: 10-JUN-89 (can be lower case)
*/
char date[];
{ int day,month,year;
  char m[4],*p;

  sscanf(date," %d-%[^-]-%d",&day,m,&year);
  for (p=m; *p; p++)
    *p = (char)toupper(*p);
  for (month=0; (month<12) && strcmp(m,MONTHS[month].m); month++)
    day += MONTHS[month].days;
  if ((month>1) && (year%4 == 0))       /* leap year and after 1-MAR    */
    day++;
  day += (year/4)*(366+3*365);
  year %= 4;
  if (year>0)
    day += 365*year + 1;
  return (day);
}

void skipCR(void)
{
  while(getchar()!='\n')
    ;
}

#ifdef ANSI
int ask(char str[],char def)
#else
int ask(str,def)
char str[],def;
#endif
{ char yes,ptr[10];

  def = (char)toupper(def);
  printf ("\n%s [Y/N D:%c]: ",str,def);
  fgets(ptr,10,stdin);
  yes = (char)toupper(ptr[0]);
  if (def=='Y')
    if (yes=='N')
      return 0;
    else
      return 1;
  else
    if (yes=='Y')
      return 1;
    else
      return 0;
}

#ifndef ANSI
char askchoice(str,def,all,va_alist)
char str[],all[], def;
va_dcl
{ char choice,tmp[10];
  va_list ptr;
  int i;

  va_start(ptr);
#else
char askchoice(char str[],char def,char all[],...)
{ char choice,tmp[10];
  va_list ptr;
  int i;

  va_start(ptr,all);
#endif
  fprintf (stderr,"\n%s",str);
  for (i=0; i<strlen(all); i++)
    fprintf (stderr,"\n\t%c : %s",all[i],va_arg(ptr,char *));
  va_end(ptr);

  do
  { fprintf (stderr,"\n\t[choice out of :%s, D:%c]: ",all,def);
    fgets(tmp,10,stdin);
    if (sscanf(tmp," %c",&choice)!=1)
      choice = def;
  } while (strchr(all,choice)==NULL);
  return(choice);
}

char *askstr(str,res,n)
char str[],res[];
{
  printf ("\n%s (max %d chars): ",str,n);
  fgets(res,n,stdin);
  if (res[strlen(res)-1]=='\n')
    res[strlen(res)-1]=0;
  else
    while (getchar()!='\n')
        ;
  return(res);
}

asknr (str,minv,maxv,def)
char str[];
int minv,maxv,def;
{ char ptr[10];
  int nnn,ret;

  while(1)
  { printf ("\n%s [%d:%d D:%d]: ",str,minv,maxv,def);
    fgets(ptr,10,stdin);
    ret=sscanf(ptr,"%d",&nnn);
    if (ret==0 || ret==EOF)
      return def;
    if ((nnn>=minv) && (nnn<=maxv))
      return nnn;
    puts("\nOut of bounds");
  }
}

long asklong (str,minv,maxv,def)
char str[];
long minv,maxv,def;
{ char ptr[10];
  int ret;
  long nnn;

  while(1)
  { printf ("\n%s [%ld:%ld D:%ld]: ",str,minv,maxv,def);
    fgets(ptr,10,stdin);
    ret=sscanf(ptr,"%ld",&nnn);
    if (ret==0 || ret==EOF)
      return def;
    if ((nnn>=minv) && (nnn<=maxv))
      return nnn;
    puts("\nOut of bounds");
  }
}

double askdouble(str,minv,maxv,def)
char str[];
double minv,maxv,def;
{ char  ptr[10];
  double nnn;
  int   ret;

  while(1)
  { printf ("\n%s (real)[%6g : %6g D:%6g]: ",str,minv,maxv,def);
    fgets(ptr,10,stdin);
    ret=sscanf(ptr,"%lf",&nnn);
    if (ret==0 || ret==EOF)
      return def;
    if ((nnn>=minv) && (nnn<=maxv))
      return nnn;
    puts("\nOut of bounds");
  }
}

char *find_filename(str)
char str[];
{  char *name;

#ifdef VAX
 name = strrchr(str,']');
 if (name==NULL)
   name = strrchr(str,':');
#elif defined(MSDOS) || ( defined(__MSL__) && __dest_os	== __win32_os )
 name = strrchr(str,'\\');
 if (name==NULL)
   name = strrchr(str,'/');
 if (name==NULL)
   name = strrchr(str,':');
#elif defined(__MSL__)  && __dest_os	== __mac_os
 name = strrchr(str,':');
#else /* Unix */
 name = strrchr(str,'/');
#endif /* VAX */
 if (name!=NULL)
   return name++;
 else
   return str;
}

char *add_extension(str,def)
char str[],def[];
{
  if (strchr(find_filename(str),'.') == NULL)
    strcat (str,def);
  return str;
}

char *asknm_extension(str,name,def)
char str[], name[], def[];
{ char ptr[300];

  name[0]='\0';
  while (strlen(name)==0)
  { printf ("\n%s (default extension %s): ",str,def);
    fgets(ptr,100,stdin);
    sscanf(ptr," %s",name);
  }
    add_extension(name,def);
    return(name);
}

FILE *askfile_extension(str,def,mode)
char str[], mode[], def[];
{ char name[300];
  FILE *f;

  for(;;)
  { asknm_extension(str,name,def);
    if ((f=fopen(name,mode))==NULL)
    { printf("\nError opening file %s",name);
      perror("\n");
      if (!ask("Retry",'N'))
        exit(EXIT_ABORT);
    }
    else
      return(f);
  }
}

/* old routines for ECAT II and IV files.
   They don't really belong here, sorry. */
char *get_pat_file(str,dir,pat)
char str[],dir[];
int  pat;
{
  strcpy(str,dir);
  sprintf(str+strlen(str),"p%05d.dat",pat);
  return str;
}

char *get_scan_name(str,dir,pat,scan)
char str[],dir[];
int  pat,scan;
{
  strcpy(str,dir);
  sprintf(str+strlen(str),"s%03d%02d", pat,scan);
  return str;
}

void get_pat_and_scan(char *fname,int *pat,int *scan)
/* fname has form saaabb where aaa = pat number
                               bb = scan number */
{ char c_pat[4],c_scan[3];
  int i;
  for (i=0;i<3;i++)
    *(c_pat+i) = *(fname+i+1);
  for (i=0;i<2;i++)
   *(c_scan+i) = *(fname+i+4);
  *pat = (int) atol(c_pat);
  *scan = (int) atol(c_scan);
}

char *get_scan_file(str,dir,pat,scan)
char str[],dir[];
int  pat,scan;
{
  return strcat(get_scan_name(str,dir,pat,scan),".i01");
}
char *get_scandat_file(str,dir,pat,scan)
char str[],dir[];
int  pat,scan;
{
  return strcat(get_scan_name(str,dir,pat,scan),".dat");
}

char *cons_dir(str,dir)
char str[],dir[];
{ int n;
  strcpy(str,dir);
  n = strlen(str) - 1;
#ifdef MSDOS
  if ((str[n]!=':') && (str[n]!='\\') && (str[n]!='/'))
    strcat(str,"\\");
#endif
#ifdef ultrix
  if (str[n]!='/')
    strcat(str,"/");
#endif
#ifdef __MSL__
#if __dest_os	== __win32_os
  if ((str[n]!=':') && (str[n]!='\\') && (str[n]!='/'))
    strcat(str,"\\");
#else
  if (str[n]!=':')
    strcat(str,":");
#endif
#endif
  return str;
}

#ifdef VAX
#include <time.h>
unsigned long CPU_time()
{
  struct tbuffer  buffer;

  times(&buffer);
  return(buffer.proc_user_time);
}
#endif
#ifdef ultrix
#include <sys/types.h>
#include <sys/times.h>
unsigned long CPU_time()
{
  struct tms buffer;

  times(&buffer);
  return((unsigned long)(10. / 6. * buffer.tms_utime));
}
#endif
#if defined(MSDOS) || defined(__MSL__)
#include <time.h>
unsigned long CPU_time()
{
  return(100*(unsigned long)time(NULL));
}
#endif

#ifndef ANSI
void message(va_alist)
va_dcl
{
  char *fmt;
  va_list ptr;

  va_start(ptr);
  fmt = va_arg(ptr, char *);
  fprintf(stderr,"\n");
  vfprintf(stderr,fmt, ptr);
  va_end(ptr);
}

void error(va_alist)
va_dcl
{ va_list ptr;
  char *fmt;

  va_start(ptr);
  fmt = va_arg(ptr, char *);
  fprintf(stderr,"\n");
  vfprintf(stderr,fmt, ptr);
  fprintf(stderr,"\n");
  va_end(ptr);
  exit(EXIT_ABORT);
}

#else
void message(char *fmt, ...)
{ va_list ptr;

  va_start(ptr,fmt);
  fprintf(stderr,"\n");
  vfprintf(stderr,fmt, ptr);
  va_end(ptr);
}

void error(char *fmt, ...)
{ va_list ptr;

  va_start(ptr,fmt);
  fprintf(stderr,"\n");
  vfprintf(stderr,fmt, ptr);
  fprintf(stderr,"\n");
  va_end(ptr);
  exit(EXIT_ABORT);
}
#endif


