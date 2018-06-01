//
//
/*!

  \file
  \ingroup Array

  \brief Implementations for class stir::SeparableMetzArrayFilter

  \author Nikos Efthimiou
  \author Matthew Jacobson
  \author Kris Thielemans
  \author Sanida Mustafovic

*/
/*
    Copyright (C) 2000 - 2009-06-22, Hammersmith Imanet Ltd
    Copyright (C) 2011-07-01 - 2011, Kris Thielemans
    Copyright (C) 2018, University of Hull
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

#include "stir/SeparableMetzArrayFilter.h"
#include "stir/ArrayFilter1DUsingConvolutionSymmetricKernel.h"
#include "stir/info.h"
#include "stir/numerics/fourier.h"
#include <boost/format.hpp>
#include <iostream>

#ifndef STIR_NO_NAMESPACES
using std::cerr;
using std::endl;
#endif

START_NAMESPACE_STIR


const float ZERO_TOL= 0.000001F; //MJ 12/05/98 Made consistent with other files
const double TPI=6.28318530717958647692;

// build Gaussian kernel according to the full width half maximum 
template <typename elemT>
static void build_gauss(VectorWithOffset<elemT>&kernel, 
			int res,float s2, float sampling_interval);

template <typename elemT>
static void build_metz(VectorWithOffset<elemT>&kernel,
		       float N,float fwhm, float MmPerVoxel, int max_kernel_size);



template <int num_dimensions, typename elemT>
SeparableMetzArrayFilter<num_dimensions,elemT>::
SeparableMetzArrayFilter
  (const VectorWithOffset<float>& fwhms_v,
   const VectorWithOffset<float>& metz_powers_v,
   const BasicCoordinate<num_dimensions,float>& sampling_distances_v,
   const VectorWithOffset<int>& max_kernel_sizes_v)
 : fwhms(fwhms_v),
   metz_powers(metz_powers_v),
   sampling_distances(sampling_distances_v),
   max_kernel_sizes(max_kernel_sizes_v)
{
  assert(metz_powers.get_length() == num_dimensions);
  assert(fwhms.get_length() == num_dimensions);
  assert(metz_powers.get_min_index() == 1);
  assert(fwhms.get_min_index() == 1);
  assert(max_kernel_sizes.get_length() == num_dimensions);
  assert(max_kernel_sizes.get_min_index() == 1);
  
  for (int i=1; i<=num_dimensions; ++i)
  {
    VectorWithOffset<elemT> kernel;
    build_metz(kernel, metz_powers[i],fwhms[i],sampling_distances[i],max_kernel_sizes[i]);
    
    for (int j=0;j<kernel.get_length();j++)
      if(metz_powers[i]>0.0)  printf ("%d-dir Metz[%d]=%f\n",i,j,kernel[j]);   
      else printf ("%d-dir Gauss[%d]=%f\n",i,j,kernel[j]);
      
      
    this->all_1d_array_filters[i-1].reset(new ArrayFilter1DUsingConvolutionSymmetricKernel<elemT>(kernel));
      
  }
}

template <typename elemT>
void build_gauss(VectorWithOffset<elemT>&kernel, int res,float s2,  float sampling_interval)
{
  
  
  elemT sum;
  int cutoff=0;
  int j,hres;
  
  
  
  hres = res/2;
  kernel[hres-1] = static_cast<elemT>(1/sqrt(s2*TPI));
  sum =   kernel[hres-1];       
  kernel[res-1] = 0;
  for (j=1;(j<hres && !cutoff);j++){
    kernel[hres-j-1] = static_cast<elemT>(kernel[hres-1]* exp(-0.5*(j*sampling_interval)*(j*sampling_interval)/s2));
    kernel[hres+j-1] = kernel [hres-j-1];
    sum +=  2.0F * kernel[hres-j-1];
    if (kernel[hres-j-1]  <kernel[hres-1]*ZERO_TOL) cutoff=1;
            
  }  
  
  /* Normalize the filter to 1 */
  for (j=0;j<res;j++) kernel[j] /= sum; 
  
}







//MJ 19/04/99  Used KT's solution to the shifted index problem. Also build_metz now allocates the kernel.
template <typename elemT>
void build_metz(VectorWithOffset<elemT>& kernel,
		float N,float fwhm, float MmPerVox, int max_kernel_size)
{    
  
  int kernel_length = 0;
  
  if(fwhm>0.0F){                                
    
    //MJ 12/05/98 compute parameters relevant to DFT/IDFT
    
    elemT s2 = fwhm*fwhm/(8*log(2.F)); //variance in Mm
    
    const int n=7; //determines cut-off in both space and frequency domains
    const elemT sinc_length=10000.0F;
    int samples_per_voxel=(int)(MmPerVox*2*sqrt(2*log(10.F)*n/s2)/TPI +1);
    const elemT sampling_interval=MmPerVox/samples_per_voxel;
    elemT stretch= (samples_per_voxel>1)?sinc_length:0.0F;
    
    int Res=(int)(log((sqrt(8*n*log(10.)*s2)+stretch)/sampling_interval)/log(2.)+1);
    Res=(int) pow(2.0,(double) Res); //MJ 12/05/98 made adaptive 
    
    info(boost::format("Filter parameters:\n"
      "Variance: %1%\n"
      "Voxel dimension (in mm): %2%\n"
      "Samples per voxel: %3%\n"
      "Sampling interval (in mm):  %4%\n"
      "FFT vector length: %5%") 
      % s2 % MmPerVox % samples_per_voxel % sampling_interval % Res);    
    
    /* allocate memory to metz arrays */
    VectorWithOffset<elemT> filter(Res);
    filter.fill(0.0);
    //The former technique was illegal.
   Array<1,std::complex<elemT> > fftdata(0,Res-1);
   fftdata.fill(0.0);
    
    /* build gaussian */
    build_gauss(filter,Res,s2,sampling_interval);

    /* Build the fft array*/
    for (int i=0;i<=Res-(Res/2);i++) {
      fftdata[i].real(filter[Res/2-1+i]);
    }
    
    for (int i=1;i<(Res/2);i++) {
      fftdata[Res-(Res/2)+i].real(filter[i-1]);
    }

    /* FFT to frequency space */
    fourier(fftdata);

    /* Build Metz */                       
    N++;
        
    int cutoff=(int) (sampling_interval*Res/(2*MmPerVox));
    //cerr<<endl<<"The cutoff was at: "<<cutoff<<endl;
    
    
    for (int i=0;i<Res;i++)
    {
     elemT xreal = fftdata[i].real();
     elemT ximg  = fftdata[i].imag();
     elemT zabs2= xreal*xreal+ximg*ximg;
	     filter[i]=0.0; // use this loop to clear the array for later
             
             
             //MJ 26/05/99 cut off the filter
             if(stretch>0.0){
               // cerr<<endl<<"truncating"<<endl;
               if(i>cutoff && Res-i>cutoff) zabs2=0;
               
             }
             
             if (zabs2>1) zabs2=static_cast<elemT> (1-ZERO_TOL);
             if (zabs2>0) {
               // if (zabs2>=1) cerr<<endl<<"zabs2 is "<<zabs2<<" and N is "<<N<<endl;
               fftdata[i].real((1-pow((1-zabs2),N))*(xreal/zabs2));
               fftdata[i].imag((1-pow((1-zabs2),N))*(-ximg/zabs2));
             }
             else {              
               fftdata[i]= 0.0;
             }
             
    }
    /* return to the spatial space */
    inverse_fourier(fftdata);

    /* collect the results, normalize*/
    
    for (int i=0;i<=Res/2;i++) {
      if (i%samples_per_voxel==0){
        int j=i/samples_per_voxel;
        filter[j] = (fftdata[i].real()*MmPerVox)/(Res*sampling_interval);
      }
      
    }
    
    
    
    //MJ 17/12/98 added step to undo zero padding (requested by RL)
    // KT 01/06/2001 added kernel_length stuff
    kernel_length=Res; 
    
    for (int i=Res-1;i>=0;i--){
      if (fabs((double) filter[i])>=(0.0001)*filter[0]) break;
      else (kernel_length)--;
      
    }
    
    
#if 0
    // SM&KT 04/04/2001 removed this truncation of the kernel as we don't have the relevant parameter anymore
    if ((kernel_length)>length_of_row_to_filter/2){
      kernel_length=length_of_row_to_filter/2;
    }
#endif

    if (max_kernel_size>0 && (kernel_length)>max_kernel_size/2){
      kernel_length=max_kernel_size/2;
    }
    
    //VectorWithOffset<elemT> kernel(kernel_length);//=new elemT[(kernel_length)];
    kernel.grow(0,kernel_length-1);
    
    for (int i=0;i<(kernel_length);i++) kernel[i]=filter[i];
    
    //return kernel;
  }
  
  else{
    //VectorWithOffset<elemT> kernel(1);//=new elemT[1];
    kernel.grow(0,0);
    //*kernel=1.0F;
    //kernel_length=1L;
    kernel[0] = 1.F;
    kernel_length=1;
    
    //return kernel;
  }

}



template class SeparableMetzArrayFilter<3, float>;

END_NAMESPACE_STIR



