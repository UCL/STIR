// $Id$

/*!
  \file 
  \ingroup test
 
  \brief tests for the conversion routines to/from VAX float used for ECAT6 support.

  \author Kris Thielemans

  $Date$
  $Revision$

*/
/*
    Copyright (C) 2004- $Date$, Hammersmith Imanet Ltd
    See STIR/LICENSE.txt for details
*/
#include "stir/ByteOrderDefine.h"
#include "stir/IO/ecat6_utils.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdarg.h>
#include <string.h>

const unsigned size = 10;
static float test_input[10] =
  { 1.F, 8.F, -1.F, .5F, 1.2345E+14F, -.12345E+6F,
    24352.245F, -34532.345F, .23457E-13F, .83561E-7F};

/* Note: this function is really almost C as the original version was
   used for the C versions of get_vax_float et al.
 */
int main(int argc, char **argv)
{
  using namespace stir;
  using namespace stir::ecat::ecat6;

  if (argc!=3)
    {
      fprintf(stderr, "Usage : --read|--write filename\n");
      exit(EXIT_FAILURE);
    }
  bool do_read;
  if (strcmp(argv[1], "--read")==0)
    do_read = true;
  else if (strcmp(argv[1], "--write")==0)
    do_read = false;
  else
    {
      fprintf(stderr, "Usage : --read|--write filename\n");
      exit(EXIT_FAILURE);
    }


  char * file_name = argv[2];
  FILE* fptr = fopen(file_name, do_read?"rb" : "wb");
  if (fptr==NULL)
    {
      fprintf(stderr, "Cannot open %s\n", file_name);
      exit(EXIT_FAILURE);
    }
  unsigned short *bufr = (unsigned short *) malloc (size*sizeof (float));
  if (!bufr) return EXIT_FAILURE;
  bool success=true;
  if (do_read)
    {
      if(!fread(bufr, /*sizeof(VAXfloat)*/4, size, fptr))
	{
	  fprintf(stderr, "Cannot read %s\n", file_name);
	  exit(EXIT_FAILURE);
	}
#if STIRIsNativeByteOrderBigEndian // we have to swap bytes in order to read ints and floats */
      swab ((char *) bufr, (char *) bufr, size*sizeof(float));
#endif
      unsigned int i;
      for ( i=0; i!=size; ++i)
	{
	  const float f= get_vax_float(bufr, 2*i);
	  if (fabs(f/test_input[i]-1)>.001)
	    {
	      fprintf(stderr,"get_vax_float error at %g, found %g\n", test_input[i], f);
	      success= false;
	    }
	}
      if (!success)
	warning("Due to failures, cannot continue with tests for hostftovaxf");
      else
	{
	  /* we do the reverse test assuming that get_vax_float works
	     Note that it's not possible to do a bit by it comparison of the
	     result of hostftovaxf due to potential rounding differences.
	  */
	  for (i=0; i!=size; ++i)
	    {
	      unsigned short local_bufr[2];
	      hostftovaxf(test_input[i], local_bufr);
	      const float f= get_vax_float(local_bufr, 0);
	      if (fabs(f/test_input[i]-1)>.001)
		{
		  fprintf(stderr,"hostftovaxf error at %g, found %g\n", test_input[i], f);
		  success= false;
		}
	    }
	}
    }
  else
    {
      unsigned int i;
      for (i=0; i!=size; ++i)
	{
	  hostftovaxf(test_input[i], (unsigned short*)(bufr+2*i));
	}
#if STIRIsNativeByteOrderBigEndian // we have to swap bytes in order to read ints and floats */
      swab ((char *) bufr, (char *) bufr, size*sizeof(float));
#endif
      if(!fwrite(bufr, /*sizeof(VAXfloat)*/4, size, fptr))
	{
	  fprintf(stderr, "Cannot write %s\n", file_name);
	  exit(EXIT_FAILURE);
	}
    }

  fflush(stderr);
  fclose (fptr);
  if (success)
    fprintf(stderr, "All tests ok\n");
  return success ? EXIT_SUCCESS : EXIT_FAILURE;
}
