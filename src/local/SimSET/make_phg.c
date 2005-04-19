	
//
// $Id$
//
/*
    Copyright (C) 2004- $Date$, Hammersmith Imanet Ltd
    This file is part of the local library STIR.

    This file is NOT free software, till University of Barcelona approves it; 
    You can NOT redistribute it and/or modify

    This file is distributed locally in the STIR library locally in the hope that it will be useful 
    for the researchers at Hammersmith Imanet Ltd.,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
*/
/*!
  \file 
  \ingroup simset
  \brief Make PHG parameter files for SimSET
  \author Pablo Aguiar
  $Date$
  $Revision$
*/

 /* This program generates a phg file to describe a object in SimSET */

#include <math.h>
#include <stdio.h> 
#include <stdlib.h>
		
#define N_INPUTS 13

void asignar_inputs(char **arg,int *nslices,int *xbins,int *ybins,float *xMin,float *xMax,float *yMin,float *yMax,float *zMin,
			float *zMax,float *zMin_target,float *zMax_target,float *radius);

int main(int argc,char **argv)
 {
int i;
double dz;              
int nslices,xbins,ybins;
float xMin,xMax,yMin,yMax,zMin,zMax,zMin_target,zMax_target,radius;

char *inputs[]={      
    "Number slices in object",
    "Number of pixels in x-direction",
    "Number of pixels in y-direction",
    "xMin (cm)",
    "xMax (cm)",
    "yMin (cm)",
    "yMax (cm)",
    "zMin (cm)",
    "zMax (cm)",
    "Scanner coordinates: zMin_target (cm)",
    "Scanner coordinates: zMax_target (cm)",
    "Scanner radius (cm)",
    };

if (argc==N_INPUTS) asignar_inputs (argv,&nslices,&xbins,&ybins,&xMin,&xMax,&yMin,&yMax,&zMin,&zMax,&zMin_target,&zMax_target,&radius);
else 
    {
    printf("\n\nmore arguments...\n");
    printf("\n\n");
    for(i=0;i<N_INPUTS-1;i++) {
        printf("\n\t%s",inputs[i]);
        }
    printf("\n\n");
    exit(0);
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
printf("\n\n# TARGET CYLINDER INFORMATION");
printf("\nNUM_ELEMENTS_IN_LIST    target_cylinder = 3");
printf("\n	REAL	target_zMin = %5.2f",zMin_target);
printf("\n	REAL	target_zMax = %5.2f",zMax_target);
printf("\n	REAL	radius =      %5.2f\n\n",radius);		
}


/*==================================== ASIGNAR_INPUTS ====================*/

void asignar_inputs(char **arg,int *nslices,int *xbins,int *ybins,float *xMin,float *xMax,float *yMin,float *yMax,float *zMin,
			float *zMax,float *zMin_target,float *zMax_target,float *radius)
{
*nslices=atoi(arg[1]);
*xbins=atoi(arg[2]);
*ybins=atoi(arg[3]);
*xMin=atof(arg[4]);
*xMax=atof(arg[5]);
*yMin=atof(arg[6]);
*yMax=atof(arg[7]);
*zMin=atof(arg[8]);
*zMax=atof(arg[9]);
*zMin_target=atof(arg[10]);
*zMax_target=atof(arg[11]);
*radius=atof(arg[12]);
}
