//
// $Id$
//

//some miscellaneous operators for sinograms and images

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>


#include "pet_common.h" 
#include "recon_array_functions.h"
#include "TensorFunction.h" 
#include <iostream> 
#include <fstream>

#include <numeric>

#define ZERO_TOL 0.000001

// KT 09/11/98 new
void truncate_rim(PETSegmentByView& seg, const int rim_truncation_sino)
{
  
  const int vs=seg.get_min_view();
  const int ve=seg.get_max_view();
  const int rs=seg.get_min_ring();
  const int re=seg.get_max_ring();
  const int bs=seg.get_min_bin();
  const int be=seg.get_max_bin();
  
  for(int v=vs;v<=ve;v++)
    for(int r=rs;r<=re;r++)
    {
      for(int b=bs;b<bs+rim_truncation_sino;b++)     
	seg[v][r][b]=0;
      for(int b=be-rim_truncation_sino+1; b<=be;b++)     
	seg[v][r][b]=0;        
    }
}

// KT 09/11/98 new
void truncate_rim(PETSegmentBySinogram& seg, const int rim_truncation_sino)
{
  
  const int vs=seg.get_min_view();
  const int ve=seg.get_max_view();
  const int rs=seg.get_min_ring();
  const int re=seg.get_max_ring();
  const int bs=seg.get_min_bin();
  const int be=seg.get_max_bin();
  
  for(int r=rs;r<=re;r++)
    for(int v=vs;v<=ve;v++)
    {
      for(int b=bs;b<bs+rim_truncation_sino;b++)     
	seg[r][v][b]=0;
      for(int b=be-rim_truncation_sino+1; b<=be;b++)     
	seg[r][v][b]=0;        
    }
}


//MJ 18/9/98 new
void truncate_rim(PETImageOfVolume& input_image,
			 const int rim_truncation_image)
{      
  const int zs=input_image.get_min_z();
  const int ys=input_image.get_min_y();
  const int xs=input_image.get_min_x(); 
  
  const int ze=input_image.get_max_z();  
  const int ye=input_image.get_max_y(); 
  const int xe=input_image.get_max_x();
  
  const int zm=(zs+ze)/2;
  const int ym=(ys+ye)/2;
  const int xm=(xs+xe)/2;
  
  const float truncated_radius = (xe-xs)/2 - rim_truncation_image;
   

  for (int z=zs; z<=ze; z++)
    for (int y=ys; y <= ye; y++)
      for (int x=xs; x<= xe; x++)
      {

	if(square(xm-x)+square(ym-y)>=square(truncated_radius))
	{
	  input_image[z][y][x]=0;
	}

      }
}

void multiply_and_add(PETImageOfVolume &image_res, const PETImageOfVolume &image_scaled, float scalar){

  int zs,ys,xs, ze,ye,xe;

  zs=image_res.get_min_z();
  ys=image_res.get_min_y();
  xs=image_res.get_min_x(); 
  
  ze=image_res.get_max_z();  
  ye=image_res.get_max_y(); 
  xe=image_res.get_max_x();

 
  for (int z=zs; z<=ze; z++)
    for (int y=ys; y <= ye; y++)
      for (int x=xs; x<= xe; x++){
           image_res[z][y][x] += image_scaled[z][y][x] *scalar;
      }

}



//to be used with in_place_function
float neg_trunc(float x){


  return (x<0.0)?0.0:x;

}

void divide_and_truncate(PETImageOfVolume& numerator, 
			 const PETImageOfVolume& denominator,
			 const int rim_truncation,
			 int & count)

{      
  const int zs=numerator.get_min_z();
  const int ys=numerator.get_min_y();
  const int xs=numerator.get_min_x(); 
  
  const int ze=numerator.get_max_z();  
  const int ye=numerator.get_max_y(); 
  const int xe=numerator.get_max_x();
  
  const int zm=(zs+ze)/2;
  const int ym=(ys+ye)/2;
  const int xm=(xs+xe)/2;
  
  const float truncated_radius = (xe-xs)/2 - rim_truncation;
   

  for (int z=zs; z<=ze; z++)
    for (int y=ys; y <= ye; y++)
      for (int x=xs; x<= xe; x++)
      {

	if(square(xm-x)+square(ym-y)>=square(truncated_radius))
	{
	  numerator[z][y][x]=0;
	}
	else
	{ 
	  if(denominator[z][y][x]<=ZERO_TOL) 
	  {
	    if(numerator[z][y][x]>ZERO_TOL) count++;	
	    numerator[z][y][x]=0;
	  }	      
	  else 
	    numerator[z][y][x]/=denominator[z][y][x];
	} 

      }
}



