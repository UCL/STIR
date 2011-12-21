/*
 $Id$
*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, Hammersmith Imanet Ltd
    See STIR/LICENSE.txt for details
*/
/*!
\file 
  
 \brief  Utility functions used by the screen* files (internal use only)
  
 \author Kris Thielemans
 \author PARAPET project
 
 $Date$
  
 $Revision$

 \see gen.h 
 \internal
*/
#include "gen.h"

#include <stdarg.h>


#ifdef VAX
#include ssdef
#include descrip
#include iodef
#include dcdef

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

#endif


int asknr (char str[],int minv,int maxv,int def)
{ char ptr[10];
  int nnn,ret;

  while(1)
  { printf ("\n%s [%d:%d D:%d]: ",str,minv,maxv,def);
    if (fgets(ptr,10,stdin) == NULL)
      return def; // nothing read, so return default
    ret=sscanf(ptr,"%d",&nnn);
    if (ret==0 || ret==EOF)
      return def;
    if ((nnn>=minv) && (nnn<=maxv))
      return nnn;
    puts("\nOut of bounds");
  }
}
size_t fread_check(char *str, void *buffer, size_t size, FILE *infile)
{ size_t ret;
  char tmp[200];

  ret = fread(buffer,sizeof(char),size,infile);
  if (ferror(infile))
  { sprintf(tmp,"\n%s: error reading for %lu bytes, %lu read\nreason",
	    str,(unsigned long)size,(unsigned long)ret);
    perror(tmp);
    error("");
  }
  return(ret);
}

size_t fwrite_check(char *str, void *buffer, size_t size, FILE *outfile)
{ size_t ret;
  char tmp[200];

  ret = fwrite(buffer,1,size,outfile);
  if (ret < size || ferror(outfile))
  { sprintf(tmp,"\n%s: error writing for %lu bytes, %lu written\nreason",
                str,(unsigned long)size,(unsigned long)ret);
    if (ferror(outfile))
      perror(tmp);
    error("");
  }
  return(ret);
}

void fseek_check(char *str, FILE *file, long int offset, int pos)
{ if (fseek(file,offset,pos))
    error("%s: error seeking in file to position %lu",str,offset);
}


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
  exit(EXIT_FAILURE);
}
