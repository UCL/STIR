/*
    Copyright (C) 2004 - 2005-04-19, Hammersmith Imanet Ltd
    Copyright (C) 2011-07-01 - 2012, Kris Thielemans
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
  \ingroup simset
  \brief Write object-spec for PHG parameter file for SimSET
  \author Pablo Aguiar
  \author Kris Thielemans
*/

#include <math.h>
#include <stdio.h> 
#include <stdlib.h>
		
#define N_INPUTS 10

/* parse cmd-line and convert to cm */
void assign_inputs(char **arg,int *nslices,int *xbins,int *ybins,
		   float *xMin,float *xMax,float *yMin,float *yMax,
		   float *zMin,	float *zMax);

int main(int argc,char **argv)
{
  int i;
  double dz;              
  int nslices,xbins,ybins;
  float xMin,xMax,yMin,yMax,zMin,zMax;

  char *inputs[]={      
    "Number slices in object",
    "Number of pixels in x-direction",
    "Number of pixels in y-direction",
    "xMin (mm)",
    "xMax (mm)",
    "yMin (mm)",
    "yMax (mm)",
    "zMin (mm)",
    "zMax (mm)"
  };

  if (argc==N_INPUTS) assign_inputs (argv,&nslices,&xbins,&ybins,&xMin,&xMax,&yMin,&yMax,&zMin,&zMax);
  else 
    {
      fprintf(stderr,"\nThis program writes the object-spec to stdout to\nhelp constructing a PHG parameter file for SimSET.\n");
      fprintf(stderr,"\nUsage:\nwrite_phg_image_info");
      for(i=0;i<N_INPUTS-1;i++) {
        fprintf(stderr," \\\n\t%s",inputs[i]);
      }
      fprintf(stderr,"\n");
      exit(EXIT_FAILURE);
    }

  dz=(zMax-zMin)/(float)nslices;
  printf("\n\n# OBJECT GEOMETRY VALUES");
  printf("\nNUM_ELEMENTS_IN_LIST	object = %d",nslices+1);
  printf("\n		INT		num_slices = %d",nslices);
  for(i=0;i<nslices;i++){
    printf("\n		NUM_ELEMENTS_IN_LIST	slice = 9 ");
    printf("\n		INT	slice_number  = %d",i);
    printf("\n		REAL	zMin = %5.2f ",zMin+((float)(i)*dz));
    printf("\n		REAL	zMax = %5.2f  ",zMin+((float)(i+1)*dz));
    printf("\n		REAL	xMin = %5.2f",xMin);
    printf("\n		REAL	xMax = %5.2f",xMax);
    printf("\n		REAL	yMin = %5.2f",yMin);
    printf("\n		REAL	yMax = %5.2f",yMax);
    printf("\n		INT	num_X_bins = %d",xbins);
    printf("\n		INT	num_Y_bins = %d\n",ybins);
  }

  return EXIT_SUCCESS;
}


/*==================================== ASSIGN_INPUTS ====================*/

void assign_inputs(char **arg,int *nslices,int *xbins,int *ybins,
		   float *xMin,float *xMax,float *yMin,float *yMax,float *zMin,float *zMax)
{
  /* parse and convert to cm */
  *nslices=atoi(arg[1]);
  *xbins=atoi(arg[2]);
  *ybins=atoi(arg[3]);
  *xMin=(float)(atof(arg[4]))/10.F;
  *xMax=(float)(atof(arg[5]))/10.F;
  *yMin=(float)(atof(arg[6]))/10.F;
  *yMax=(float)(atof(arg[7]))/10.F;
  *zMin=(float)(atof(arg[8]))/10.F;
  *zMax=(float)(atof(arg[9]))/10.F;
}
