//
// $Id$
//
/*!
  \file
  \ingroup buildblock
  \brief implementations for functions declared in recon_array_functions.h

  \author Matthew Jacobson
  \author Kris Thielemans
  \author PARAPET project
  
  $Date$

  $Revision$
*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/

//some miscellaneous operators for sinograms and images

#include "stir/recon_array_functions.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/RelatedViewgrams.h"
#include "stir/Viewgram.h"
#include "stir/SegmentByView.h"
#include "stir/SegmentBySinogram.h"

#include <numeric>

#ifndef STIR_NO_NAMESPACES
using std::cout;
using std::cerr;
using std::endl;
#endif

START_NAMESPACE_STIR

const float SMALL_NUM = 0.000001F;

void truncate_min_to_small_positive_value(DiscretisedDensity<3,float>& input_image, const int rim_truncation_image){


  long nuneg;
  float Minneg;

  nuneg=0;
  Minneg = 0.0;

  // TODO the 'rim_truncate' part of this function does not make a lot of sense in general
  DiscretisedDensityOnCartesianGrid<3,float>& input_image_cartesian =
    dynamic_cast<DiscretisedDensityOnCartesianGrid<3,float>&>(input_image);

  const int zs=input_image_cartesian.get_min_z();
  const int ys=input_image_cartesian.get_min_y();
  const int xs=input_image_cartesian.get_min_x(); 
  
  const int ze=input_image_cartesian.get_max_z();  
  const int ye=input_image_cartesian.get_max_y(); 
  const int xe=input_image_cartesian.get_max_x();
  
  // TODO check what happens with even-sized images (i.e. where is the centre?)
 
  const int ym=(ys+ye)/2;
  const int xm=(xs+xe)/2;


  float small_value=min_positive_value(input_image,rim_truncation_image);
  small_value=(small_value>0.F)?small_value*SMALL_NUM:SMALL_NUM;

  const float truncated_radius = (xe-xs)/2 - rim_truncation_image;


  for (int z=zs; z<=ze; z++)
    for (int y=ys; y <= ye; y++)
      for (int x=xs; x<= xe; x++)
	{

	//MJ 12/104/2000 '<0.0' ---> '<=0.0'and limit to FOV
	//KT&MJ 23/05/2000 changed comparison with 0 to small_value
	//MJ 13/07/2000 changed it back, equivalent under new thresholding strategy 
	if(square(xm-x)+square(ym-y)<=square(truncated_radius) && 
	   input_image[z][y][x]<=0.F)
	  {
	    if (input_image[z][y][x]<Minneg) Minneg = input_image[z][y][x];
	    nuneg++;       
	    input_image[z][y][x]=small_value; 
       
	  }

	//MJ optimal implementation would disregard voxels outside FOV
	else if (square(xm-x)+square(ym-y)>square(truncated_radius))
	  input_image[z][y][x]=0.0;
	
	}

  cout << nuneg <<" negative numbers found\n";
  cout <<"Minimal negative number " << Minneg << endl;
  cout << "All negative numbers were set to: " << small_value << endl;

}


// AZ 07/10/99: added
void truncate_rim(Viewgram<float>& viewgram, const int rim_truncation_sino)
{
  const int rs=viewgram.get_min_axial_pos_num();
  const int re=viewgram.get_max_axial_pos_num();
  const int bs=viewgram.get_min_tangential_pos_num();
  const int be=viewgram.get_max_tangential_pos_num();
  
 //MJ 12/04/2000 to remove the necessity for grow in things like sum_over_projections()
  int upper_truncation_offset=rim_truncation_sino;
  if(viewgram.get_num_tangential_poss()%2!=0) upper_truncation_offset--;


  for(int r=rs;r<=re;r++)
    {
      for(int b=bs;b<bs+rim_truncation_sino;b++)     
	viewgram[r][b]=0;
      for(int b=be-upper_truncation_offset; b<=be;b++)     
	viewgram[r][b]=0;        
    }
}

void truncate_rim(SegmentByView<float>& seg, const int rim_truncation_sino)
{
  
  const int vs=seg.get_min_view_num();
  const int ve=seg.get_max_view_num();
  const int rs=seg.get_min_axial_pos_num();
  const int re=seg.get_max_axial_pos_num();
  const int bs=seg.get_min_tangential_pos_num();
  const int be=seg.get_max_tangential_pos_num();
 
  //MJ 25/03/2000 to remove the necessity for grow in things like sum_over_projections()
  int upper_truncation_offset=rim_truncation_sino;
  if(seg.get_num_tangential_poss()%2!=0) upper_truncation_offset--;

  for(int v=vs;v<=ve;v++)
    for(int r=rs;r<=re;r++)
    {
      for(int b=bs;b<bs+rim_truncation_sino;b++)     
	seg[v][r][b]=0;
      for(int b=be-upper_truncation_offset; b<=be;b++)     
	seg[v][r][b]=0;        
    }

}


void truncate_rim(SegmentBySinogram<float>& seg, const int rim_truncation_sino)
{
  
  const int vs=seg.get_min_view_num();
  const int ve=seg.get_max_view_num();
  const int rs=seg.get_min_axial_pos_num();
  const int re=seg.get_max_axial_pos_num();
  const int bs=seg.get_min_tangential_pos_num();
  const int be=seg.get_max_tangential_pos_num();
  
  //MJ 25/03/2000 to remove the necessity for grow in things like sum_over_projections()
  int upper_truncation_offset=rim_truncation_sino;
  if(seg.get_num_tangential_poss()%2!=0) upper_truncation_offset--;


  for(int r=rs;r<=re;r++)
    for(int v=vs;v<=ve;v++)
    {
      for(int b=bs;b<bs+rim_truncation_sino;b++)     
	seg[r][v][b]=0;
      for(int b=be-upper_truncation_offset;b<=be;b++)     
	seg[r][v][b]=0;        
    }
}

//MJ 18/9/98 new
void truncate_rim(DiscretisedDensity<3,float>& input_image,
			 const int rim_truncation_image)
{      
  // TODO the 'rim_truncate' part of this function does not make a lot of sense in general
  DiscretisedDensityOnCartesianGrid<3,float>& input_image_cartesian =
    dynamic_cast<DiscretisedDensityOnCartesianGrid<3,float>&>(input_image);

  const int zs=input_image_cartesian.get_min_z();
  const int ys=input_image_cartesian.get_min_y();
  const int xs=input_image_cartesian.get_min_x(); 
  
  const int ze=input_image_cartesian.get_max_z();  
  const int ye=input_image_cartesian.get_max_y(); 
  const int xe=input_image_cartesian.get_max_x();
  
  // TODO check what happens with even-sized images (i.e. where is the centre?)
  
  //const int zm=(zs+ze)/2;
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




// AZ&KT 04/10/99: added rim_truncation_sino
void divide_and_truncate(Viewgram<float>& numerator, 
			 const Viewgram<float>& denominator,
			 const int rim_truncation_sino,
			 int& count, int& count2, float* log_likelihood_ptr /* = NULL */)
{
  
  const int rs=numerator.get_min_axial_pos_num();
  const int re=numerator.get_max_axial_pos_num();
  const int bs=numerator.get_min_tangential_pos_num();
  const int be=numerator.get_max_tangential_pos_num();


  const float small_value= 
    max(numerator.find_max()*SMALL_NUM, 0.F);
  
  for(int r=rs;r<=re;r++)
    for(int b=bs;b<=be;b++){      
 
      // KT&SM&MJ 21/05/2001 changed truncation strategy
      // before singularities (non-zero divided by zero) were set to 0
      // now they are set to max_quotient
#if 0
      // old version
      if(denominator[r][b]<=small_value ||
	 numerator[r][b]<=0.0 ||
	 b<bs+rim_truncation_sino ||
	 b>be-rim_truncation_sino ) {
	if(numerator[r][b]>small_value && denominator[r][b]<=small_value) count++;
	else if( numerator[r][b]<0.0) count2++;
	numerator[r][b]=0.0;
      }     
      else {
	//MJ 28/10/99 corrected - moved above the sinogram division
	if (log_likelihood_ptr != NULL) {
	      *log_likelihood_ptr -= numerator[r][b]*log(denominator[r][b]);
	};
	numerator[r][b]/=denominator[r][b];
      };

#else
      if(b<bs+rim_truncation_sino ||
	 b>be-rim_truncation_sino ) 
	{
	  numerator[r][b] = 0;
	}
      else
	{
	  float& num = numerator[r][b];
	  if (num<small_value)
	    { 
	      // we think num was really 0 
	      // (we compare with small_value due to rounding errors)
	      // this case includes 0/0, but also num<0	     
	      num = 0;
	      if (num<0) count2++;
	    }
	  else
	    {
	      const float max_quotient = 10000.F;
	      const float denom = denominator[r][b];
	      // set quotient to min(numerator/denominator, max_quotient)
	      // a bit tricky to avoid division by 0	  
	      // we do this by effectively using
	      // new_denom = max(denominator[r][b], max_quotient/num)
	      // Note that this includes the case if a negative denominator
	      // (in case somebody forward projects an image with negatives)
	      if (num > max_quotient*denom)
		{
		  // cancel singularity
		  count++;
		  if (log_likelihood_ptr != NULL) 
		    *log_likelihood_ptr -= num*log(num/max_quotient);
		  num = max_quotient;
		}
	      else
		{
		  if (log_likelihood_ptr != NULL) 
		    *log_likelihood_ptr -= num*log(denom);
		  num = num/denom;
		}
	    }
	}
#endif
    }

}



void divide_and_truncate(RelatedViewgrams<float>& numerator, const RelatedViewgrams<float>& denominator,
			 const int rim_truncation_sino,
			 int& count, int& count2, float* log_likelihood_ptr )
{
  assert(numerator.get_num_viewgrams() == denominator.get_num_viewgrams());
  assert(*(numerator.get_proj_data_info_ptr()) == (*denominator.get_proj_data_info_ptr()));
  RelatedViewgrams<float>::iterator numerator_iter = numerator.begin();
  RelatedViewgrams<float>::const_iterator denominator_iter = denominator.begin();
  while(numerator_iter!=numerator.end())
  {
   divide_and_truncate(*numerator_iter,
                        *denominator_iter,
                        rim_truncation_sino,
                        count, count2, log_likelihood_ptr);
  ++numerator_iter;
  ++denominator_iter;
  }
}


void divide_and_truncate(DiscretisedDensity<3,float>& numerator, 
			 const DiscretisedDensity<3,float>& denominator,
			 const int rim_truncation,
			 int & count)
{      
   assert(numerator.get_index_range() == denominator.get_index_range());

  // TODO the 'truncate' part of this function does not make a lot of sense in general
  DiscretisedDensityOnCartesianGrid<3,float>& numerator_cartesian =
    dynamic_cast<DiscretisedDensityOnCartesianGrid<3,float>&>(numerator);

  const int zs=numerator_cartesian.get_min_z();
  const int ys=numerator_cartesian.get_min_y();
  const int xs=numerator_cartesian.get_min_x(); 
  
  const int ze=numerator_cartesian.get_max_z();  
  const int ye=numerator_cartesian.get_max_y(); 
  const int xe=numerator_cartesian.get_max_x();
  
  // TODO check what happens with even-sized images (i.e. where is the centre?)
  //const int zm=(zs+ze)/2;
  const int ym=(ys+ye)/2;
  const int xm=(xs+xe)/2;
  
  const float truncated_radius = (xe-xs)/2 - rim_truncation;

  float small_value= (float) numerator.find_max()*SMALL_NUM;
  small_value=(small_value>0.0)?small_value:0.0;   

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
	  //MJ 20/6/2000 use fabs(). 
	  if(fabs(denominator[z][y][x])<=small_value) 
	  {
	    if(fabs(numerator[z][y][x])>small_value) count++;	
	    numerator[z][y][x]=0;
	  }	      
	  else 
	    numerator[z][y][x]/=denominator[z][y][x];
	} 

      }

}


void divide_array(SegmentByView<float>& numerator,const SegmentByView<float>& denominator)
{
  
  const int vs=numerator.get_min_view_num();
  const int ve=numerator.get_max_view_num();
  const int rs=numerator.get_min_axial_pos_num();
  const int re=numerator.get_max_axial_pos_num();
  const int bs=numerator.get_min_tangential_pos_num();
  const int be=numerator.get_max_tangential_pos_num();
  
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

void divide_array(DiscretisedDensity<3,float>& numerator, const DiscretisedDensity<3,float>& denominator)
{
    assert(numerator.get_index_range() == denominator.get_index_range());
  // TODO rewrite in terms of 'full' iterator
 
  for (int z=numerator.get_min_index(); z<=numerator.get_max_index(); z++)
    for (int y=numerator[z].get_min_index(); y<=numerator[z].get_max_index(); y++)
      for (int x=numerator[z][y].get_min_index(); x<=numerator[z][y].get_max_index(); x++)
      { 

	if (denominator[z][y][x] != 0.0)
	  numerator[z][y][x]=numerator[z][y][x]/denominator[z][y][x];
	else
	  numerator[z][y][x]=0.0;

      }
    
}

//  MJ 03/01/2000 for loglikelihood computation
// KT 21/05/2001 make sure it returns same result as divide_and_truncate above
void accumulate_loglikelihood(Viewgram<float>& projection_data, 
			 const Viewgram<float>& estimated_projections,
			 const int rim_truncation_sino,
			 float* accum)
{
  
  const int rs=projection_data.get_min_axial_pos_num();
  const int re=projection_data.get_max_axial_pos_num();
  const int bs=projection_data.get_min_tangential_pos_num();
  const int be=projection_data.get_max_tangential_pos_num();

  
  const float small_value= 
    max(projection_data.find_max()*SMALL_NUM, 0.F);
  const float max_quotient = 10000.F;

  for(int r=rs;r<=re;r++)
    for(int b=bs;b<=be;b++)  
      if(!(
	   projection_data[r][b]<=small_value ||
	   b<bs+rim_truncation_sino ||
	   b>be-rim_truncation_sino ))
	{
	  const float new_estimate =
	    max(estimated_projections[r][b], 
		max_quotient/projection_data[r][b]);
	  *accum -= projection_data[r][b]*log(new_estimate);
	}

}



void multiply_and_add(DiscretisedDensity<3,float> &image_res, const DiscretisedDensity<3,float> &image_scaled, float scalar)
{

  assert(image_res.get_index_range() == image_scaled.get_index_range());
  // TODO rewrite in terms of 'full' iterator
 
  for (int z=image_res.get_min_index(); z<=image_res.get_max_index(); z++)
    for (int y=image_res[z].get_min_index(); y<=image_res[z].get_max_index(); y++)
      for (int x=image_res[z][y].get_min_index(); x<=image_res[z][y].get_max_index(); x++)
	{
           image_res[z][y][x] += image_scaled[z][y][x] *scalar;
	}

}



//to be used with in_place_function
float neg_trunc(float x)
{

  return (x<0.0)?0.0:x;

}



void truncate_end_planes(DiscretisedDensity<3,float> &input_image, int input_num_planes)
{

  // TODO this function does not make a lot of sense in general
#ifndef NDEBUG
  // this will throw an exception when the cast is invalid
  dynamic_cast<DiscretisedDensityOnCartesianGrid<3,float>&>(input_image);
#endif

  const int zs=input_image.get_min_index();
  const int ze=input_image.get_max_index();

  int upper_limit=(input_image.get_length() % 2 == 1)?input_image.get_length()/2+1:input_image.get_length()/2;

  int num_planes=input_num_planes<=upper_limit?input_num_planes:upper_limit;

 for (int j=0;j<num_planes;j++ )
   {
     input_image[zs+j].fill(0.0);
     input_image[ze-j].fill(0.0);
   }


}

//MJ 13/07/2000 new
//TODO template? generalize to other Arrays?

float min_positive_value(DiscretisedDensity<3,float>& input_image, const int rim_truncation_image)
{


  DiscretisedDensityOnCartesianGrid<3,float>& input_image_cartesian =
    dynamic_cast<DiscretisedDensityOnCartesianGrid<3,float>&>(input_image);

  const int zs=input_image_cartesian.get_min_z();
  const int ys=input_image_cartesian.get_min_y();
  const int xs=input_image_cartesian.get_min_x(); 
  
  const int ze=input_image_cartesian.get_max_z();  
  const int ye=input_image_cartesian.get_max_y(); 
  const int xe=input_image_cartesian.get_max_x();
  
  const int ym=(ys+ye)/2;
  const int xm=(xs+xe)/2;

  const float truncated_radius = (xe-xs)/2 - rim_truncation_image;

  float current_estimate=0.F;

  bool found_positive_value=false;

  for (int z=zs; z<=ze; z++)
    for (int y=ys; y <= ye; y++)
      for (int x=xs; x<= xe; x++)
	{
    
	  if(square(xm-x)+square(ym-y)<=square(truncated_radius))
	    {

	      if(found_positive_value)
		{
		  if(input_image[z][y][x]>0.F && input_image[z][y][x]<current_estimate)
		    current_estimate=input_image[z][y][x];
		}

	      else if (input_image[z][y][x]>0.F)
		{

		found_positive_value=true;
		current_estimate=input_image[z][y][x];

		}

	    }

	}

  return current_estimate;
	

}




END_NAMESPACE_STIR
