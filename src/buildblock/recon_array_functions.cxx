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

void divide_and_truncate_den(const PETImageOfVolume& numerator, 
			 PETImageOfVolume& denominator,
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
	  denominator[z][y][x]=0;
	}
	else
	{ 
	  if(denominator[z][y][x]<=ZERO_TOL) 
	  {
	    if(numerator[z][y][x]>ZERO_TOL) count++;	
	    denominator[z][y][x]=0;
	  }	      
	  else 
	    denominator[z][y][x]=numerator[z][y][x]/denominator[z][y][x];
	} 

      }


}

// AZ&KT 04/10/99: removed view45, added rim_truncation_sino
void divide_and_truncate(const int view, // const int view45,
			 PETSegmentBySinogram& numerator, 
			 const PETSegmentBySinogram& denominator,
			 const int rim_truncation_sino,
			 int& count, int& count2, float* f /* = NULL */)
{
  assert(numerator.get_num_views() == denominator.get_num_views());

  const int view45 = numerator.get_num_views() / 4;

  const int view90 = view45*2;
  const int plus90 = view90+view;
  const int min180 = view45*4-view;
  const int min90  = view90-view;
	
  int v[4],max_index;

  if (view != 0 /* && view != globals.view45 */ ){
    max_index=3;
    v[0]=view;v[1]=plus90;v[2]=min90;v[3]=min180;
  }
	
  else{
    max_index=3;
    v[0]=view;v[1]=plus90;
    v[2]=view45;v[3]=view90+view45; 
  }

  const int rs=denominator.get_min_ring();
  const int re=denominator.get_max_ring();
  const int vs=denominator.get_min_view();
  const int ve=denominator.get_max_view();
  const int bs=denominator.get_min_bin();
  const int be=denominator.get_max_bin();

  for(int r=rs;r<=re;r++)
    for(int i=0;i<=max_index;i++)
      for(int b=bs;b<=be;b++) {
	      
	if(denominator[r][v[i]][b]<=ZERO_TOL ||/* numerator[r][v[i]][b]/denominator[r][v[i]][b]>100.0 ||*/
	   numerator[r][v[i]][b]<0.0|| 
	   b<bs+rim_truncation_sino || b>be-rim_truncation_sino) {
	  if(numerator[r][v[i]][b]>ZERO_TOL && denominator[r][v[i]][b]<=ZERO_TOL ) count++;
	  else if( numerator[r][v[i]][b]<0.0) count2++;
	  numerator[r][v[i]][b]=0.0;
	}	      
	else {

	  //MJ 28/10/99 corrected - moved above the sinogram division
	  if (f != NULL) {
	    *f -= numerator[r][v[i]][b]*log(denominator[r][v[i]][b]); //Check the validity of this
	  };

	  numerator[r][v[i]][b]/=denominator[r][v[i]][b];

	
	};
      }

  //	if (f != NULL)
  //  cerr<<endl<<"f="<<*f<<endl;
	
}

// AZ&KT 04/10/99: added rim_truncation_sino
void divide_and_truncate(PETViewgram& numerator, 
			 const PETViewgram& denominator,
			 const int rim_truncation_sino,
			 int& count, int& count2, float* f /* = NULL */)
{
  
  const int rs=numerator.get_min_ring();
  const int re=numerator.get_max_ring();
  const int bs=numerator.get_min_bin();
  const int be=numerator.get_max_bin();
  
  for(int r=rs;r<=re;r++)
    for(int b=bs;b<=be;b++){      
      
      if(denominator[r][b]<=ZERO_TOL ||/* numerator[r][b]/denominator[r][b]>100.0 ||*/
	 numerator[r][b]<0.0 ||
	 b<bs+rim_truncation_sino ||
	 b>be-rim_truncation_sino ) {
	if(numerator[r][b]>ZERO_TOL && denominator[r][b]<=ZERO_TOL ) count++;
	else if( numerator[r][b]<0.0) count2++;
	numerator[r][b]=0.0;
      }
      
      else {

	//MJ 28/10/99 corrected - moved above the sinogram division
	if (f != NULL) {
	      *f -= numerator[r][b]*log(denominator[r][b]);
	};

	numerator[r][b]/=denominator[r][b];


      };
    }

  //	if (f != NULL)
  //  cerr<<endl<<"f="<<*f<<endl;

}

// AZ 07/10/99: added
void truncate_rim(PETSegmentBySinogram& seg, const int rim_truncation_sino, const int view)
{
  const int rs=seg.get_min_ring();
  const int re=seg.get_max_ring();
  const int bs=seg.get_min_bin();
  const int be=seg.get_max_bin();
  
  for(int r=rs;r<=re;r++)
    {
      for(int b=bs;b<bs+rim_truncation_sino;b++)     
	seg[r][view][b]=0;
      for(int b=be-rim_truncation_sino+1; b<=be;b++)     
	seg[r][view][b]=0;        
    }
}

// AZ 07/10/99: added
void truncate_rim(PETViewgram& viewgram, const int rim_truncation_sino)
{
  const int rs=viewgram.get_min_ring();
  const int re=viewgram.get_max_ring();
  const int bs=viewgram.get_min_bin();
  const int be=viewgram.get_max_bin();
  
  for(int r=rs;r<=re;r++)
    {
      for(int b=bs;b<bs+rim_truncation_sino;b++)     
	viewgram[r][b]=0;
      for(int b=be-rim_truncation_sino+1; b<=be;b++)     
	viewgram[r][b]=0;        
    }
}


void truncate_end_planes(PETImageOfVolume &input_image)
{

const int zs=input_image.get_min_z();
const int ze=input_image.get_max_z();

input_image[zs].fill(0.0);
input_image[ze].fill(0.0);



}



void divide_array(PETSegmentByView& numerator,PETSegmentByView& denominator)
{
  
  const int vs=numerator.get_min_view();
  const int ve=numerator.get_max_view();
  const int rs=numerator.get_min_ring();
  const int re=numerator.get_max_ring();
  const int bs=numerator.get_min_bin();
  const int be=numerator.get_max_bin();
  
  for(int v=vs;v<=ve;v++)
    for(int r=rs;r<=re;r++)
      for(int b=bs;b<=be;b++)
	{

	  if (denominator[v][r][b] != 0.0)
	    numerator[v][r][b]=numerator[v][r][b]/denominator[v][r][b];
	  else
	     numerator[v][r][b]=0.0;

	}
    
}



// MJ 03/01/2000 for loglikelihood computation
void accumulate_loglikelihood(const int view,
			 PETSegmentBySinogram& projection_data, 
			 const PETSegmentBySinogram& estimated_projections,
			 const int rim_truncation_sino, float *accum)
{


  assert(projection_data.get_num_views() == estimated_projections.get_num_views());


  const int view45 = projection_data.get_num_views() / 4;

  const int view90 = view45*2;
  const int plus90 = view90+view;
  const int min180 = view45*4-view;
  const int min90  = view90-view;
	
  int v[4],max_index;

  if (view != 0 /* && view != globals.view45 */ ){
    max_index=3;
    v[0]=view;v[1]=plus90;v[2]=min90;v[3]=min180;
  }
	
  else{
    max_index=3;
    v[0]=view;v[1]=plus90;
    v[2]=view45;v[3]=view90+view45; 
  }

  const int rs=estimated_projections.get_min_ring();
  const int re=estimated_projections.get_max_ring();
  const int vs=estimated_projections.get_min_view();
  const int ve=estimated_projections.get_max_view();
  const int bs=estimated_projections.get_min_bin();
  const int be=estimated_projections.get_max_bin();

  for(int r=rs;r<=re;r++)
    for(int i=0;i<=max_index;i++)
      for(int b=bs;b<=be;b++)	      
	if(!(estimated_projections[r][v[i]][b]<=ZERO_TOL ||
	   projection_data[r][v[i]][b]<0.0|| b<bs+rim_truncation_sino 
	   || b>be-rim_truncation_sino))
	  *accum -= projection_data[r][v[i]][b]*log(estimated_projections[r][v[i]][b]);

  // cerr<<endl<<"accum="<<*accum<<endl;

}

//  MJ 03/01/2000 for loglikelihood computation
void accumulate_loglikelihood(PETViewgram& projection_data, 
			 const PETViewgram& estimated_projections,
			 const int rim_truncation_sino,
			 float* accum)
{
  
  const int rs=projection_data.get_min_ring();
  const int re=projection_data.get_max_ring();
  const int bs=projection_data.get_min_bin();
  const int be=projection_data.get_max_bin();
  
  for(int r=rs;r<=re;r++)
    for(int b=bs;b<=be;b++)  
      if(!(estimated_projections[r][b]<=ZERO_TOL ||
	 projection_data[r][b]<0.0 ||
	 b<bs+rim_truncation_sino ||
	 b>be-rim_truncation_sino ))
	      *accum -= projection_data[r][b]*log(estimated_projections[r][b]);
    
  // cerr<<endl<<"accum="<<*accum<<endl;

}
