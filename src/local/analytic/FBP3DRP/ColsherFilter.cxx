//
// $Id$
//

/*! 
  \file 
  \brief Colsher filter implementation
  \author Claire LABBE
  \author Kris Thielemans
  \author based on C-code by Matthias Egger
  \author PARAPET project
  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/


#include <math.h>

#include "stir/Segment.h"
#include "stir/Viewgram.h"

#include "local/stir/FBP3DRP/ColsherFilter.h"
// TODO remove
#include <iostream>
#include <fstream>
#ifdef BOOST_NO_STRINGSTREAM
#include <strstream.h>
#else
#include <sstream>
#endif

//#include "stir/interfile.h"

#ifndef STIR_NO_NAMESPACE
using std::ends;
using std::ofstream;
using std::cout;
using std::endl;
#endif

START_NAMESPACE_STIR

static const double epsilon = 1e-10;


string ColsherFilter::parameter_info() const
{
#ifdef BOOST_NO_STRINGSTREAM
  // dangerous for out-of-range, but 'old-style' ostrstream seems to need this
  char str[1000];
  ostrstream s(str, 1000);
#else
  std::ostringstream s;
#endif  
    s << "\nPETFilterColsherParameters :="
      << "\nFilter height := "<< height
      << "\nFilter width := "<< width      
      << "\nCut-off in cycles along planar direction:= "<< fc_planar
      << "\nCut-off in cycles along axial direction:= "<< fc_axial
      << "\nAlpha parameter along planar direction := "<<alpha_planar
      << "\nAlpha parameter along axial direction:= "<<alpha_axial
      << "\nRadial sampling (along bin direction) := "<< d_a
      << "\nAxial sampling (along ring direction) := "<< d_b
      << "\nMaximum aperture (theta_max) := "<< theta_max
      << "\nCopolar angle (gamma):= "<< gamma
      << ends;
    
    return s.str();
}



ColsherFilter::ColsherFilter(int height_v, int width_v, float gamma_v, float theta_max_v,
				 float d_a_v, float d_b_v,
                                 float alpha_colsher_axial_v, float fc_colsher_axial_v,
                                 float alpha_colsher_planar_v, float fc_colsher_planar_v) 
        : Filter2D<float>(height_v, width_v),gamma(gamma_v), theta_max(theta_max_v), d_a(d_a_v), d_b(d_b_v),
          alpha_axial(alpha_colsher_axial_v), fc_axial(fc_colsher_axial_v),
          alpha_planar(alpha_colsher_planar_v), fc_planar(fc_colsher_planar_v)
{
 
  if (height==0 || width==0)
    return;

    int             k, j;
    float           fa, fb, omega, psi;
    float           fil, mod_nu, nu_a, nu_b;
	/*
	 * The Colsher filter is real-valued, so it has only height*width elements,
	 * going from [1..height*width]. It is arranged in wrap-around order
	 * in both dimensions, see Num.Rec.C, page 523
	 */
 
    int ii = 1;//float fmax = 0.F;
	
    // TODO suspicious (?) that loop does something for height==0
    for (j = 0; j <= 0.5 * height; j++) {
        fb = (float) j / height;
        nu_b = fb / d_b;
        for (k = 0; k <= 0.5 * width; k++) {
            fa = (float) k / width;
            nu_a = fa / d_a;
            if (fa == 0. && fb == 0.) {
                filter[ii++] = 0.;
                continue;
            }

            omega = atan2(nu_b, nu_a);
            psi = acos(sin(omega) * sin(gamma));
            mod_nu = sqrt(nu_a * nu_a + nu_b * nu_b);

            /* Colsher formula = fil = mod_nu / 4. / asin(sin(theta_max) / sin(psi)); */
            if (cos(psi) >= cos(theta_max))
                fil = mod_nu / 2. / _PI;
            else
                fil = mod_nu / 4. / asin(sin(theta_max) / sin(psi));
                      
                       
                /* Apodizing Hanning window */;
                //CL 250899 In order to make simial than the CTI program, we use a damping window
                //for both planar and axial direction

            if (fa < fc_planar || fb < fc_axial) 
                filter[ii++] = fil * (alpha_planar + (1. - alpha_planar) * cos(_PI * fa / fc_planar))
                    *(alpha_axial + (1. - alpha_axial)* cos(_PI * fb / fc_axial));
            else
                filter[ii++] = 0.;

        }
                
               
        for (k = int (-0.5 * width) + 1; k <= -1; k++) {
            fa = (float) k / width;
            nu_a = fa / d_a;
            omega = atan2(nu_b, -nu_a);
            psi = acos(sin(omega) * sin(gamma));
               
                        
            mod_nu = sqrt(nu_a * nu_a + nu_b * nu_b);

            if (cos(psi) >= cos(theta_max))
                fil = mod_nu / 2. / _PI;
            else
                fil = mod_nu / 4. / asin(sin(theta_max) / sin(psi));


            if (-fa < fc_planar || fb < fc_axial) 
                filter[ii++] = fil * (alpha_planar + (1. - alpha_planar) * cos(_PI * (-fa) / fc_planar))
                    *(alpha_axial + (1. - alpha_axial)* cos(_PI * fb / fc_axial));
            else
                filter[ii++] = 0.;                        
		
        }
    }
        
    for (j = int (-0.5 * height) + 1; j <= -1; j++) {
        fb = (float) j / height;
        nu_b = fb / d_b;
        for (k = 0; k <= 0.5 * width; k++) {
            fa = (float) k / width;
            nu_a = fa / d_a;
            omega = atan2(-nu_b, nu_a);
            psi = acos(sin(omega) * sin(gamma));
              
            mod_nu = sqrt(nu_a * nu_a + nu_b * nu_b);

            if (cos(psi) >= cos(theta_max))
                fil = mod_nu / 2. / _PI;
            else
                fil = mod_nu / 4. / asin(sin(theta_max) / sin(psi));

 
            if (fa < fc_planar || -fb < fc_axial) 
                filter[ii++] = fil * (alpha_planar + (1. - alpha_planar) * cos(_PI * fa / fc_planar))
                    *(alpha_axial + (1. - alpha_axial)* cos(_PI * (-fb) / fc_axial));
            else
                filter[ii++] = 0.;
                    
        }
                
        for (k = int (-0.5 * width) + 1; k <= -1; k++) {
            fa = (float) k / width;
            nu_a = fa / d_a;
            omega = atan2(-nu_b, -nu_a);
            psi = acos(sin(omega) * sin(gamma));
              
            mod_nu = sqrt(nu_a * nu_a + nu_b * nu_b);

            if (cos(psi) >= cos(theta_max))
                fil = mod_nu / 2. / _PI;
            else
                fil = mod_nu / 4. / asin(sin(theta_max) / sin(psi));

            if (-fa < fc_planar || -fb < fc_axial) 
                filter[ii++] = fil * (alpha_planar + (1. - alpha_planar) * cos(_PI * (-fa) / fc_planar))
                    *(alpha_axial + (1. - alpha_axial)* cos(_PI * (-fb) / fc_axial));
            else
                filter[ii++] = 0.;

                         
        }
    }

    // KT&Darren Hogg 03/07/2001 inserted correct scale factor 
    // TODO this assumes current value for the magic_number in bakcprojector
    filter *= 4*_PI*d_a;

    
    {
            
        if(0){
            char file[200];
            sprintf(file,"%s_%d_%d.dat","filtColsher",width,height);
            cout << "Saving filter : " << file << endl;
    
            ofstream logfile(file);
 
            if (logfile.fail() || logfile.bad()) {
                error("Error opening file\n");
            }
            for(int p= 1; p <= height*width; p++)
                logfile << filter[p] << " ";
        }
    }
        
            
}

#if 0

/*!
  filter all views
  \warning This routines breaks when the number of views is odd.

  TODONEW add error checking for that case <BR>
  TODONEW rmin, rmax should really come from segment, gamma also
*/
static void Filter_proj_Colsher(Segment<float>& segment, int i, 
                                float gamma, float theta_max, float d_a, float d_b,
                                float alpha_colsher_axial, float fc_colsher_axial,
                                float alpha_colsher_planar, float fc_colsher_planar,
                                int PadS, int PadZ); 
#endif



#ifndef PARALLEL

//CL24/02/00 AS the function without rmin and rmax, and the one with these two variables
// are quite similar, remove the content of the function without rmin, and rmax, 
// by calling the other function
void Filter_proj_Colsher(Segment<float> & segment, 
                         float gamma, float theta_max, float d_a, float d_b,
                         float alpha_colsher_axial, float fc_colsher_axial,
                         float alpha_colsher_planar, float fc_colsher_planar,
                         int PadS, int PadZ)
{

    Filter_proj_Colsher(segment, gamma, theta_max, d_a, d_b,
                        alpha_colsher_axial,  fc_colsher_axial,
                        alpha_colsher_planar,  fc_colsher_planar,
                        PadS,  PadZ,
                        segment.get_min_axial_pos_num(), segment.get_max_axial_pos_num());
    
}

#if 0
static void Filter_proj_Colsher(Segment<float> &segment, int i, 
                                float gamma, float theta_max, float d_a, float d_b,
                                float alpha_colsher_axial, float fc_colsher_axial,
                                float alpha_colsher_planar, float fc_colsher_planar,
                                int PadS, int PadZ, int rmin, int rmax);
#endif

void Filter_proj_Colsher(Segment<float> &segment, 
                         float gamma, float theta_max, float d_a, float d_b,
                         float alpha_colsher_axial, float fc_colsher_axial,
                         float alpha_colsher_planar, float fc_colsher_planar,
                         int PadS, int PadZ, int rmin, int rmax)
{
    //start_timers();

    int nrings = rmax - rmin + 1; 

    int nprojs = segment.get_num_tangential_poss();

    int width = (int) pow(2, ((int) ceil(log((PadS + 1) * nprojs) / log(2))));
    int height = (int) pow(2, ((int) ceil(log((PadZ + 1) * nrings) / log(2))));	
      
    const int maxproj = segment.get_max_tangential_pos_num();
    const int minproj = segment.get_min_tangential_pos_num();

    int roffset = -rmin * width *2; 
    Array<1,float> data(1,2 * height * width);

       
    ColsherFilter Cfilter(height, width, gamma, theta_max, d_a, d_b,
                            alpha_colsher_axial, fc_colsher_axial,
                            alpha_colsher_planar, fc_colsher_planar);
       

    for (int i = segment.get_min_view_num(); i <= segment.get_max_view_num(); i += 2)
    {
	data.fill(0.F);

	Viewgram<float> view_i = segment.get_viewgram(i);
	Viewgram<float> view_i1 = segment.get_viewgram(i+1);
    	{
            for (int j = rmin; j <= rmax; j++) 
		for (int k = minproj; k <= maxproj; k++) {
                    data[roffset + 2 * j * width + 2 * (k - minproj) + 1] = view_i[j][k];
                    data[roffset + 2 * j * width + 2 * (k - minproj) + 2] = view_i1[j][k];
                }
	}
 
	Cfilter.apply(data);
   
           
	{
            for (int j = rmin; j <= rmax; j++) 
		for (int k = minproj; k <= maxproj; k++) {
                    view_i[j][k] = data[roffset + 2 * j * width + 2 * (k - minproj) + 1];
                    view_i1[j][k] =data[roffset + 2 * j * width + 2 * (k - minproj) + 2];
                }
	}
	segment.set_viewgram(view_i);
	segment.set_viewgram(view_i1);
    }
     
    //stop_timers();
}

//filter all views
#else  // PARALLEL code 

#error empty for now

#endif  // PARALLEL

void Filter_proj_Colsher(Viewgram<float> & view_i,
			 Viewgram<float> & view_i1,
                         float theta, float theta0, float d_a, float d_b,
                         float alpha_colsher_axial, float fc_colsher_axial,
                         float alpha_colsher_planar, float fc_colsher_planar,
                         int PadS, int PadZ, int rmin, int rmax)
{
  int nrings = rmax - rmin + 1; 
  int nprojs = view_i.get_num_tangential_poss();
  
  int width = (int) pow(2, ((int) ceil(log((PadS + 1) * nprojs) / log(2))));
  int height = (int) pow(2, ((int) ceil(log((PadZ + 1) * nrings) / log(2))));	
  

  ColsherFilter Cfilter(height, width, theta, theta0, d_a, d_b,
      alpha_colsher_axial, fc_colsher_axial,
      alpha_colsher_planar, fc_colsher_planar);
 
  Filter_proj_Colsher(view_i, view_i1, Cfilter, PadS, PadZ);

}

void Filter_proj_Colsher(Viewgram<float> & view_i,
			 Viewgram<float> & view_i1,
                         ColsherFilter& CFilter, 
                         int PadS, int PadZ)
{  
  // start_timers();
  
  const int rmin = view_i.get_min_axial_pos_num();
  const int rmax = view_i.get_max_axial_pos_num();
  int nrings = rmax - rmin + 1; 
  int nprojs = view_i.get_num_tangential_poss();
  
  int width = (int) pow(2, ((int) ceil(log((PadS + 1) * nprojs) / log(2))));
  int height = (int) pow(2, ((int) ceil(log((PadZ + 1) * nrings) / log(2))));	
  
  const int maxproj = view_i.get_max_tangential_pos_num();
  const int minproj = view_i.get_min_tangential_pos_num();
  
  int roffset = -rmin * width *2; 
  Array<1,float> data(1,2 * height * width);
  
  
  for (int j = rmin; j <= rmax; j++) 
  {
    for (int k = minproj; k <= maxproj; k++) {
      data[roffset + 2 * j * width + 2 * (k - minproj) + 1] = view_i[j][k];
      data[roffset + 2 * j * width + 2 * (k - minproj) + 2] = view_i1[j][k];
    }
  }
  {
    
    CFilter.apply(data);
  }
  
  
  
  
  for (int j = rmin; j <= rmax; j++) 
  {
    for (int k = minproj; k <= maxproj; k++) 
    {
      view_i[j][k] = data[roffset + 2 * j * width + 2 * (k - minproj) + 1];
      view_i1[j][k] =data[roffset + 2 * j * width + 2 * (k - minproj) + 2];
    }
  }
  
  // stop_timers();
  
}



template <class T>
void zoom_filter(Array<1,T> &filter, int height, int width, int height_proj, int width_proj)
{
  
  // Transform the 1D filter to 2D matrix for simplification
  int ii=1;
  int h,w,x1,y1,x2,y2;
  
  //AS the Fourier properties said
  //if h is even , H is even
  
  if(height_proj%2==1)
    height_proj++;
  if(width_proj%2==1)
    width_proj++;
  
  Array<2,float> filter2D(1,height,1,width);//-height/2 +1,height/2, -width/2 +1, width/2);
  Array<2,float> filter2D_scale(1,height_proj,1,width_proj);
  
  
  for (h = 1 ; h <= height; h++) 
    for (w = 1; w <= width; w++) 
      filter2D[h][w] = filter[ii++];
    
    char file[200];
    if(0)  {
      sprintf(file,"%s","filt2D.dat");
      
      ofstream logfile(file);
      
      if (logfile.fail() || logfile.bad()) {
        PETerror("Error opening file\n");
        Abort();
      }
      for(int h=1; h <= height; h++)
        for(int p=1; p <= width; p++)
          logfile << filter2D[h][p] << " ";
    }
    //  filter2D.set_offsets(1,1);
    
    double zoom_axial= (double) height/height_proj;
    double pix1_ax= 1;
    double pix2_ax= zoom_axial;
    
    double zoom_radial= (double) width/width_proj;
    double pix1_rad= 1;
    double pix2_rad= zoom_radial;
    
    
    double dx = 1, dy = 1;
    
    y2=1;
    for (y1= filter2D.get_h_min(); y1 <= filter2D.get_h_max(); y1++){
      
      if(y2> filter2D_scale.get_length2())
        break;
      x2=1;
      for (x1= filter2D.get_w_min(); x1 <= filter2D.get_w_max();  x1++){
        
        if(x2> filter2D_scale.get_length1())
          break;
        // x,y are located somewhere within a grid of 4 voxels
        if (x1*pix1_rad - x2*pix2_rad <= 0)
          dx=1;
        else if(!(x1*pix1_rad - x2*pix2_rad <= 0))
          dx= 1- (x1*pix1_rad - x2*pix2_rad)/pix1_rad;
        if (y1*pix1_ax - y2*pix2_ax <= 0)
          dy=1;
        else if (!(y1*pix1_ax - y2*pix2_ax <= 0))
          dy= 1- (y1*pix1_ax - y2*pix2_ax)/pix1_ax;
        
        filter2D_scale[y2 + filter2D_scale.get_h_min()-1][x2 + filter2D_scale.get_w_min()-1] +=
          (filter2D[y1][x1]*dx*dy);
        
        if(dx!=1 && dy!=1)
          filter2D_scale[y2 + filter2D_scale.get_h_min()][x2 + filter2D_scale.get_w_min()] +=
          (filter2D[y1][x1]*(1-dx)*(1-dy));
        
        if (dx!=1)
          filter2D_scale[y2 + filter2D_scale.get_h_min()-1][x2 + filter2D_scale.get_w_min()] +=
          (filter2D[y1][x1]*(1-dx)*dy);
        if(dy!=1)
          filter2D_scale[y2 + filter2D_scale.get_h_min()][x2 + filter2D_scale.get_w_min()-1] +=
          (filter2D[y1][x1]*dx*(1-dy));
        // CL due to rounding values, test on x1*pix1 -x2*pix2==0 is not done
        if  ((fabs(x1*pix1_rad -x2*pix2_rad) < 0) || dx!=1) 
          x2++;
        
      }// End of for x2
      if ((fabs(y1*pix1_ax -y2*pix2_ax) < 0) || dy!=1)
        y2++;
    }// End of for y2
    
    
    //ZEROES PADDING
    
    
    filter2D.fill(0);
    
    //PADDING
    //   x2=0;//(int)(-0.5 * width);//filter2D.get_min_index1()-1;
    y2=1;//(int)(-0.5 * height); // filter2D.get_min_index2()-1;
    
    for (y1= filter2D_scale.get_h_min(); y1 <= filter2D_scale.get_h_max(); y1++){
      x2=1;
      
      for (x1= filter2D_scale.get_w_min(); x1 <= filter2D_scale.get_w_max();  x1++){
        
        filter2D[y2][x2] = filter2D_scale[y1][x1]/zoom_radial/zoom_axial;
        
        if(x1==filter2D_scale.get_w_max()/2)
          x2 = filter2D.get_w_max()-filter2D_scale.get_w_max()/2;
        x2++;
      }
      if (y1==filter2D_scale.get_h_max()/2)
        y2 = filter2D.get_h_max()-filter2D_scale.get_h_max()/2;
      y2++;
    }
    
    
    if(0)  {
      sprintf(file,"%s","filtzoom.dat");
      
      ofstream logfile(file);
      
      if (logfile.fail() || logfile.bad()) {
        PETerror("Error opening file\n");
        Abort();
      }
      for(int h=1; h <= height; h++)
        for(int p=1; p <= width; p++)
          logfile << filter2D[h][p] << " ";
    }
    // Rearrange 2D matrix to a 1D array
    
    //  filter2D.set_offsets((int)(-0.5 * height) + 1,(int)(-0.5 * width) + 1);
    ii=1;
    
    for (h = 1 ; h <= height; h++) 
      for (w = 1; w <= width; w++) 
        filter[ii++]=filter2D[h][w];
      
}



//CL 011298 New module for solving the problem of non axial uniformity
// due to the appropriate padding in the filter
template <class T> void Filter2D<T>::padd_scale_filter(int height_proj, int width_proj)
{
  //cout<<" Padding with zeroes in Colsher filter from "
  //<< height_proj <<" to " << height << " and from "
  //<< width_proj  <<" to " << width  << endl
  char file[200];
  if(0)    {
    sprintf(file,"%s","filt1.dat");
    
    ofstream logfile(file);
    
    if (logfile.fail() || logfile.bad()) {
      PETerror("Error opening file\n");
      Abort();
    }
    for(int p=1; p <= height*width; p++)
      logfile << filter[p] << " ";
  }
  
  // Inverse FFT of the filter which is in Fourier space (or frequential space)
  // to have the filter in domain space
  
  Array<1,unsigned long>  nn(1,2);
  nn[1] = height;
  nn[2] = width;
  
  Array<1,float> filter_tmp(1,2*height*width);
  
  for (int j = 1, k = 1; j < width * height + 1; j++) {
    filter_tmp[k++] = filter[j];// Only fill the Real part
    k++;
  }
  
  
  fourn(filter_tmp, nn, 2, -1);
  //CL10/03/00 Add a multiplicative factor by the product of the lengths of all dimensions
  filter_tmp/= nn.sum();
  
  //Put it back to filter
  for (int j = 1, k = 1; j < width * height + 1; j++) {
    filter[j]=filter_tmp[k++];
    k++;
  }
  
  
  
  if(0) {
    sprintf(file,"%s","filt1z.dat");
    
    ofstream logfile(file);
    
    if (logfile.fail() || logfile.bad()) {
      error("Error opening file\n");
    }
    for(int p=1; p <= height*width; p++)
      logfile << filter[p] << " ";
  }
  
  
  // Scaling the original length of the filter to the original length of bins (i.e num_bins)
  //  zoom_filter(filter,height,width,height_proj, width_proj);
  
  if(0){
    sprintf(file,"%s","filt2z.dat");
    
    ofstream logfile(file);
    
    if (logfile.fail() || logfile.bad()) {
      error("Error opening file\n");
    }
    for(int p=1; p <= height*width; p++)
      logfile << filter[p] << " ";
  }
  for (int j = 1, k = 1; j < width * height + 1; j++) {
    filter_tmp[k++] = filter[j];
    k++;
  }
  // Zeroes padding by extending the filter to the original length of the filter
  // this means that a large contiguous section of the data, in the middle of
  // that array, is zero, with non-zeros values clustered at the two extreme
  // ends of the array
  //For the moment padding is done in zoom_filter
  //   padding(filter,height,width,height_proj, width_proj);
  // FFT of this modified filter to get back to the Fourier space
  
  fourn(filter_tmp, nn, 2, 1);
  for (int j = 1, k = 1; j < width * height + 1; j++) {
    filter[j]=filter_tmp[k++];
    k++;
  }
  
  if(0) {
    sprintf(file,"%s","filt2.dat");
    
    ofstream logfile(file);
    
    if (logfile.fail() || logfile.bad()) {
      error("Error opening file\n");
    }
    for(int p=1; p <= height*width; p++)
      logfile << filter[p] << " ";
  }
  
}


END_NAMESPACE_STIR
