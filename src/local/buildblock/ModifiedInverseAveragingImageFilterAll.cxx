//
/*!

  \file
  \ingroup buildblock
  \brief Implementations for class ModifiedInverseAveragingImageFilterAll
  
    \author Sanida Mustafovic
    \author Kris Thielemans
    
*/
/*
    Copyright (C) 2000- 2003, IRSL
    See STIR/LICENSE.txt for details
*/
#include "local/stir/ModifiedInverseAveragingImageFilterAll.h"
#include "stir/IndexRange3D.h"
#include "stir/ProjDataFromStream.h"
#include "stir/IO/interfile.h"
#include "stir/CartesianCoordinate3D.h"
#include "stir/RelatedViewgrams.h"
#include "local/stir/fft.h"
#include "local/stir/ArrayFilter2DUsingConvolution.h"
#include "local/stir/recon_buildblock/ProjMatrixByDenselUsingRayTracing.h"
#include "stir/CPUTimer.h"
#include "stir/SegmentByView.h"

#include "stir/round.h"
#include <iostream>
#include <fstream>
#include <algorithm>

#ifndef STIR_NO_NAMESPACES
using std::ios;
using std::find;
using std::iostream;
using std::fstream;
using std::cerr;
using std::endl;
#endif

#include "local/stir/local_helping_functions.h"
#include "local/stir/fwd_and_bck_manipulation_for_SAF.h"



START_NAMESPACE_STIR
	

void
construct_scaled_filter_coefficients_3D(VectorWithOffset < VectorWithOffset < VectorWithOffset < float> > > &new_filter_coefficients_3D_array,   
				     VectorWithOffset<float> kernel_1d,
				     const float kapa0_over_kapa1);

void
construct_scaled_filter_coefficients_2D(Array<2,float> &new_filter_coefficients_2D_array,   
				     VectorWithOffset<float> kernel_1d,
				     const float kapa0_over_kapa1);
				     
		  //// IMPLEMENTATION /////
/**********************************************************************************************/



void
construct_scaled_filter_coefficients_3D(Array<3,float> &new_filter_coefficients_3D_array,   
				     VectorWithOffset<float> kernel_1d,
				     const float kapa0_over_kapa1)
				     
{
  
  // in the case where sq_kappas=1 --- scaled_filter == original template filter 
  Array<3,float> filter_coefficients(IndexRange3D(kernel_1d.get_min_index(), kernel_1d.get_max_index(),
    kernel_1d.get_min_index(), kernel_1d.get_max_index(),
    kernel_1d.get_min_index(), kernel_1d.get_max_index()));
  
  create_kernel_3d (filter_coefficients, kernel_1d);
  
  
#if 1
  // STUFF FOR THE FFT SIZE 
  /****************** *********************************************************************/
  
  const int length_of_size_array = 16;
  const float kapa0_over_kapa1_interval_size=10.F;
  static VectorWithOffset<int> size_for_kapa0_over_kapa1;
  if (size_for_kapa0_over_kapa1.get_length()==0)
  {
    size_for_kapa0_over_kapa1.grow(0,length_of_size_array-1);
    size_for_kapa0_over_kapa1.fill(64);
  }
  
  const int kapa0_over_kapa1_interval = 
    min(static_cast<int>(floor(kapa0_over_kapa1/kapa0_over_kapa1_interval_size)),
    length_of_size_array-1);
  
  float sq_kapas = kapa0_over_kapa1; 
  /******************************************************************************************/
  
  if ( sq_kapas > 10000)
  {
    new_filter_coefficients_3D_array.grow(IndexRange3D(0,0,0,0,0,0));
  }
  else if (sq_kapas!=1.F)
  {
    
    while(true)
    {
      const int size = size_for_kapa0_over_kapa1[kapa0_over_kapa1_interval];
      
      int filter_length = static_cast<int>(floor(kernel_1d.get_length()/2));
      
      //cerr << "Now doing size " << size << std::endl;
      
      // FIRST PADD 1D FILTER COEFFICIENTS AND MAKE THEM SYMMETRIC 
      // ( DO NOT TAKE IMAGINARY PART INTO ACCOUNT YET)
      /**********************************************************************************/
      
      Array<1,float> filter_coefficients_padded_1D_array(1,size);
      
      filter_coefficients_padded_1D_array[1] = kernel_1d[0];  
      for ( int i = 1;i<=filter_length;i++)
      {
	filter_coefficients_padded_1D_array[i+1] = kernel_1d[i];    
	filter_coefficients_padded_1D_array[size-(i-1)] = kernel_1d[i];
	
      }
      
      /*************************************************************************************/
      
      VectorWithOffset < float> kernel_1d_vector;
      kernel_1d_vector.grow(1,size);
      for ( int i = 1; i<= size ;i++)
	kernel_1d_vector[i] = filter_coefficients_padded_1D_array[i];
      
      Array<3,float> filter_coefficients_padded(IndexRange3D(1,size,1,size,1,size));
      
      create_kernel_3d (filter_coefficients_padded, kernel_1d_vector);
      
      
      // rescale to DC=1
      filter_coefficients_padded /= filter_coefficients_padded.sum();
      
      Array<3,float>& fft_filter = filter_coefficients_padded;
      
      float inverse_sq_kapas;
      if (fabs((double)sq_kapas ) >0.000000000001)
	inverse_sq_kapas = 1/sq_kapas;
      else 
	inverse_sq_kapas = 0;
      
      
     // Array<1,float> fft_1_1D_array (1, 2*fft_filter.get_length()*fft_filter[fft_filter.get_min_index()].get_length() *fft_filter[fft_filter.get_min_index()][fft_filter.get_min_index()].get_length());
      Array<1,float> fft_filter_1D_array(1, 2*fft_filter.get_length()*fft_filter[fft_filter.get_min_index()].get_length() *fft_filter[fft_filter.get_min_index()][fft_filter.get_min_index()].get_length());
      
      static shared_ptr <Array<1,float> > fft_filter_1D_array_64 = 
	new Array<1,float>(1,2*64*64*64);
      static shared_ptr <Array<1,float> > fft_filter_1D_array_128 =
	new  Array<1,float> (1,2*128*128*128);
      static shared_ptr <Array<1,float> > fft_filter_1D_array_256 = 
	new  Array<1,float> (1,2*256*256*256);
      
      
      convert_array_3D_into_1D_array(fft_filter_1D_array,fft_filter);
     // fft_1_1D_array[1]=1;
      
      Array<1, int> array_lengths(1,3);
      array_lengths[1] = fft_filter.get_length();
      array_lengths[2] = fft_filter[fft_filter.get_min_index()].get_length();
      array_lengths[3] = fft_filter[fft_filter.get_min_index()][fft_filter.get_min_index()].get_length();
      Array<3,float> new_filter_coefficients_3D_array_tmp (IndexRange3D(1,filter_coefficients_padded.get_length(),1,filter_coefficients_padded.get_length(),1,filter_coefficients_padded.get_length()));
      
      // initialise to 0 to prevent from warnings
      float  fft_1_1D_array  = 0; 
      //fourn(fft_1_1D_array, array_lengths, 3,1); 
      
      if (size == 64)
      {
	if ( (*fft_filter_1D_array_64)[1] == 0.F)
	{
	  fourn(fft_filter_1D_array, array_lengths, 3,1);
	  fft_filter_1D_array /= sqrt(static_cast<double>(size *size*size));      
	  *fft_filter_1D_array_64 = fft_filter_1D_array;
	}
	else
	{
	  fft_filter_1D_array = *fft_filter_1D_array_64;

	}
      }
      else if (size ==128)
      {
	if ( (*fft_filter_1D_array_128)[1] == 0.F)
	{
	  fourn(fft_filter_1D_array, array_lengths, 3,1);
          fft_filter_1D_array /= sqrt(static_cast<double>(size *size*size));      
	  *fft_filter_1D_array_128 = fft_filter_1D_array;
	}
	else
	{
	  fft_filter_1D_array = *fft_filter_1D_array_128;

	}
      }
      else if ( size == 256)
      {
	if ( (*fft_filter_1D_array_256)[1] == 0.F)
	{
	  fourn(fft_filter_1D_array, array_lengths, 3,1);
          fft_filter_1D_array /= sqrt(static_cast<double>(size *size*size));      
	  *fft_filter_1D_array_256 = fft_filter_1D_array;
	}
	else
	{
	  fft_filter_1D_array = *fft_filter_1D_array_256;

	}
      }
      else     
      {
	warning("\nModifiedInverseAveragingImageFilter: Cannot do this at the moment -- size is too big'.\n");
	
      }
      
     // fourn(fft_filter_1D_array, array_lengths, 3,1);
      
      // WARNING  -- this only works for the FFT where the convention is that the final result
      // obtained from the FFT is divided with sqrt(N*N*N)
      switch (size)
      {
      case 64:
	fft_1_1D_array = static_cast<float>(1/sqrt(static_cast<float>(64*64*64)));
	break;
      case 128:
	fft_1_1D_array = static_cast<float>(1/sqrt(static_cast<float>(128*128*128)));
	break;
      case 256:
	fft_1_1D_array = static_cast<float>(1/sqrt(static_cast<float>(256*256*256)));
	break;
      
      default:
	warning("\nModifiedInverseAveragingImageFilter: Cannot do this at the moment -- size is too big'.\n");;
	break;
      }
      
      // to check the outputs make the fft consistant with mathematica
      // divide 1/sqrt(size)
      //fft_1_1D_array /= sqrt(static_cast<double> (size *size*size));
     // fft_filter_1D_array /= sqrt(static_cast<double>(size *size*size));
      
      {        
	Array<1,float> fft_filter_num_1D_array(1, 2*fft_filter.get_length()*fft_filter[fft_filter.get_min_index()].get_length() *fft_filter[fft_filter.get_min_index()][fft_filter.get_min_index()].get_length());
	//Array<1,float> div_1D_array(1, 2*fft_filter.get_length()*fft_filter[fft_filter.get_min_index()].get_length() *fft_filter[fft_filter.get_min_index()][fft_filter.get_min_index()].get_length());    
	
	//mulitply_complex_arrays(fft_filter_num_1D_array,fft_filter_1D_array,fft_1_1D_array);
	
	fft_filter_num_1D_array = fft_filter_1D_array* fft_1_1D_array;
	for ( int k = fft_filter_1D_array.get_min_index(); k<=fft_filter_1D_array.get_max_index();k++)
	{
	  fft_filter_1D_array[k] *= (sq_kapas-1);
	  fft_filter_1D_array[k] += fft_1_1D_array;
	  fft_filter_1D_array[k] /= sq_kapas;
	  
	}
	
	//divide_complex_arrays(div_1D_array,fft_filter_num_1D_array,fft_filter_1D_array);        
	
	divide_complex_arrays(fft_filter_num_1D_array,fft_filter_1D_array);              
	fourn(fft_filter_num_1D_array, array_lengths,3,-1);
	
	// make it consistent with mathemematica -- the output of the       
	fft_filter_num_1D_array  /= sqrt(static_cast<double>(size *size*size));
	
	
#if 0
	
	cerr << "for kappa_0_over_kappa_1 " << kapa0_over_kapa1 ;
	for (int i =1;i<=size*size*size;i++)
	{
	  cerr << fft_filter_num_1D_array[i] << "   ";
	}  
#endif
	
	// take the real part only 
	/*********************************************************************************/
	{
	  Array<1,float> real_div_1D_array(1,fft_filter.get_length()*fft_filter[fft_filter.get_min_index()].get_length() *fft_filter[fft_filter.get_min_index()][fft_filter.get_min_index()].get_length());
	  
	  for (int i=0;i<=(size*size*size)-1;i++)
	    real_div_1D_array[i+1] = fft_filter_num_1D_array[2*i+1];
	  
	  /*********************************************************************************/
	 	  
	  
	  convert_array_1D_into_3D_array(new_filter_coefficients_3D_array_tmp,real_div_1D_array);
	}
      }
	  int kernel_length_x=0;
	  int kernel_length_y=0;
	  int kernel_length_z=0;
	  
	  // to prevent form aliasing limit the new range for the coefficients to 
	  // filter_coefficients_padded.get_length()/4
	  
	  
	  // do the x -direction first -- fix y and z to the min and look for the max index in x
	  int kx = new_filter_coefficients_3D_array_tmp.get_min_index();
	  int jx = new_filter_coefficients_3D_array_tmp[kx].get_min_index();
	  for (int i=new_filter_coefficients_3D_array_tmp[kx][jx].get_min_index();i<=filter_coefficients_padded[kx][jx].get_max_index()/4;i++)
	  {
	    if (fabs((double)new_filter_coefficients_3D_array_tmp[kx][jx][i])
	      <= new_filter_coefficients_3D_array_tmp[new_filter_coefficients_3D_array_tmp.get_min_index()][new_filter_coefficients_3D_array_tmp[new_filter_coefficients_3D_array_tmp.get_min_index()].get_min_index()][new_filter_coefficients_3D_array_tmp[new_filter_coefficients_3D_array_tmp.get_min_index()][new_filter_coefficients_3D_array_tmp[new_filter_coefficients_3D_array_tmp.get_min_index()].get_min_index()].get_min_index()]*1/1000000) break;
	    else (kernel_length_x)++;
	  }
	  
	  /******************************* Y DIRECTION ************************************/
	  
	  
	  int ky = new_filter_coefficients_3D_array_tmp.get_min_index();
	  int iy = new_filter_coefficients_3D_array_tmp[ky][new_filter_coefficients_3D_array_tmp.get_min_index()].get_min_index();
	  for (int j=new_filter_coefficients_3D_array_tmp[ky].get_min_index();j<=filter_coefficients_padded[ky].get_max_index()/4;j++)
	  {
	    if (fabs((double)new_filter_coefficients_3D_array_tmp[ky][j][iy])
	      //= new_filter_coefficients_3D_array_tmp[new_filter_coefficients_3D_array_tmp.get_min_index()][new_filter_coefficients_3D_array_tmp.get_min_index()][new_filter_coefficients_3D_array_tmp.get_min_index()]*1/100000000) break;
	      <= new_filter_coefficients_3D_array_tmp[new_filter_coefficients_3D_array_tmp.get_min_index()][new_filter_coefficients_3D_array_tmp[new_filter_coefficients_3D_array_tmp.get_min_index()].get_min_index()][new_filter_coefficients_3D_array_tmp[new_filter_coefficients_3D_array_tmp.get_min_index()][new_filter_coefficients_3D_array_tmp[new_filter_coefficients_3D_array_tmp.get_min_index()].get_min_index()].get_min_index()]*1/1000000) break;
	    else (kernel_length_y)++;
	  }
	  
	  /********************************************************************************/
	  
	  /******************************* z DIRECTION ************************************/
	  
	  
	  int jz = new_filter_coefficients_3D_array_tmp.get_min_index();
	  int iz = new_filter_coefficients_3D_array_tmp[jz][new_filter_coefficients_3D_array_tmp.get_min_index()].get_min_index();
	  for (int k=new_filter_coefficients_3D_array_tmp.get_min_index();k<=filter_coefficients_padded.get_max_index()/4;k++)
	  {
	    if (fabs((double)new_filter_coefficients_3D_array_tmp[k][jz][iz])
	      //<= new_filter_coefficients_3D_array_tmp[new_filter_coefficients_3D_array_tmp.get_min_index()][new_filter_coefficients_3D_array_tmp.get_min_index()][new_filter_coefficients_3D_array_tmp.get_min_index()]*1/100000000) break;
	      <= new_filter_coefficients_3D_array_tmp[new_filter_coefficients_3D_array_tmp.get_min_index()][new_filter_coefficients_3D_array_tmp[new_filter_coefficients_3D_array_tmp.get_min_index()].get_min_index()][new_filter_coefficients_3D_array_tmp[new_filter_coefficients_3D_array_tmp.get_min_index()][new_filter_coefficients_3D_array_tmp[new_filter_coefficients_3D_array_tmp.get_min_index()].get_min_index()].get_min_index()]*1/1000000) break;
	    else (kernel_length_z)++;
	  }
	  
	  /********************************************************************************/
	  
	  if (kernel_length_x == filter_coefficients_padded.get_length()/4)
	  {
	    warning("ModifiedInverseAverigingArrayFilter3D: kernel_length_x reached maximum length %d. "
	      "First filter coefficient %g, last %g, kappa0_over_kappa1 was %g\n"
	      "Increasing length of FFT array to resolve this problem\n",
	      kernel_length_x, new_filter_coefficients_3D_array_tmp[new_filter_coefficients_3D_array_tmp.get_min_index()][new_filter_coefficients_3D_array_tmp.get_min_index()][new_filter_coefficients_3D_array_tmp.get_min_index()], new_filter_coefficients_3D_array_tmp[new_filter_coefficients_3D_array_tmp.get_min_index()][new_filter_coefficients_3D_array_tmp.get_min_index()][kernel_length_x],
	      kapa0_over_kapa1);
	    size_for_kapa0_over_kapa1[kapa0_over_kapa1_interval]*=2;
	    for (int i=kapa0_over_kapa1_interval+1; i<size_for_kapa0_over_kapa1.get_length(); ++i)
	      size_for_kapa0_over_kapa1[i]=
	      max(size_for_kapa0_over_kapa1[i], size_for_kapa0_over_kapa1[kapa0_over_kapa1_interval]);
	  }
	  else if (kernel_length_y == filter_coefficients_padded.get_length()/4)
	  {
	    warning("ModifiedInverseAverigingArrayFilter3D: kernel_length_y reached maximum length %d. "
	      "First filter coefficient %g, last %g, kappa0_over_kappa1 was %g\n"
	      "Increasing length of FFT array to resolve this problem\n",
	      kernel_length_y, new_filter_coefficients_3D_array_tmp[new_filter_coefficients_3D_array_tmp.get_min_index()][new_filter_coefficients_3D_array_tmp.get_min_index()][new_filter_coefficients_3D_array_tmp.get_min_index()], new_filter_coefficients_3D_array_tmp[new_filter_coefficients_3D_array_tmp.get_min_index()][kernel_length_y][new_filter_coefficients_3D_array_tmp.get_min_index()],
	      kapa0_over_kapa1);
	    size_for_kapa0_over_kapa1[kapa0_over_kapa1_interval]*=2;
	    for (int i=kapa0_over_kapa1_interval+1; i<size_for_kapa0_over_kapa1.get_length(); ++i)
	      size_for_kapa0_over_kapa1[i]=
	      max(size_for_kapa0_over_kapa1[i], size_for_kapa0_over_kapa1[kapa0_over_kapa1_interval]);
	  }
	  else if (kernel_length_z == filter_coefficients_padded.get_length()/4)
	  {
	    warning("ModifiedInverseAverigingArrayFilter3D: kernel_length_z reached maximum length %d. "
	      "First filter coefficient %g, last %g, kappa0_over_kappa1 was %g\n"
	      "Increasing length of FFT array to resolve this problem\n",
	      kernel_length_z, 
	      new_filter_coefficients_3D_array_tmp[new_filter_coefficients_3D_array_tmp.get_min_index()][new_filter_coefficients_3D_array_tmp.get_min_index()][new_filter_coefficients_3D_array_tmp.get_min_index()], new_filter_coefficients_3D_array_tmp[kernel_length_z][new_filter_coefficients_3D_array_tmp.get_min_index()][new_filter_coefficients_3D_array_tmp.get_min_index()],
	      kapa0_over_kapa1);
	    size_for_kapa0_over_kapa1[kapa0_over_kapa1_interval]*=2;
	    for (int i=kapa0_over_kapa1_interval+1; i<size_for_kapa0_over_kapa1.get_length(); ++i)
	      size_for_kapa0_over_kapa1[i]=
	      max(size_for_kapa0_over_kapa1[i], size_for_kapa0_over_kapa1[kapa0_over_kapa1_interval]);
	  }
	  else
	  {	 
	    new_filter_coefficients_3D_array.grow(IndexRange3D(-(kernel_length_z-1),kernel_length_z,
	      -(kernel_length_y-1),kernel_length_y,
	      -(kernel_length_x-1),kernel_length_x));
	    
	    new_filter_coefficients_3D_array[0][0][0] = new_filter_coefficients_3D_array_tmp[1][1][1];
	    new_filter_coefficients_3D_array[kernel_length_z][kernel_length_y][kernel_length_x] = new_filter_coefficients_3D_array_tmp[kernel_length_z][kernel_length_y][kernel_length_x];
	    
	    
	    for (int  k = 1;k<= kernel_length_z-1;k++)
	      for (int  j = 1;j<= kernel_length_y-1;j++)	  
		for (int  i = 1;i<= kernel_length_x-1;i++)	  
		  
		{
		  new_filter_coefficients_3D_array[k][j][i]=new_filter_coefficients_3D_array_tmp[k+1][j+1][i+1];
		  new_filter_coefficients_3D_array[-k][-j][-i]=new_filter_coefficients_3D_array_tmp[k+1][j+1][i+1];
		  
		}
		
		break; // out of while(true)
	  }
    } // this bracket is for the while loop
    }
    else //sq_kappas == 1
    {
      new_filter_coefficients_3D_array = filter_coefficients;
    }
    
    
#endif 
    
           
    // rescale to DC=1
    float sum_new_coefficients =0;  
    for (int k=new_filter_coefficients_3D_array.get_min_index();k<=new_filter_coefficients_3D_array.get_max_index();k++)
      for (int j=new_filter_coefficients_3D_array[k].get_min_index();j<=new_filter_coefficients_3D_array[k].get_max_index();j++)	  
	for (int i=new_filter_coefficients_3D_array[k][j].get_min_index();i<=new_filter_coefficients_3D_array[k][j].get_max_index();i++)
	  sum_new_coefficients += new_filter_coefficients_3D_array[k][j][i];  
	
	for (int k=new_filter_coefficients_3D_array.get_min_index();k<=new_filter_coefficients_3D_array.get_max_index();k++)
	  for (int j=new_filter_coefficients_3D_array[k].get_min_index();j<=new_filter_coefficients_3D_array[k].get_max_index();j++)	  
	    for (int i=new_filter_coefficients_3D_array[k][j].get_min_index();i<=new_filter_coefficients_3D_array[k][j].get_max_index();i++)
	      new_filter_coefficients_3D_array[k][j][i] /=sum_new_coefficients;  
	

		    
}

void
construct_scaled_filter_coefficients_2D(Array<2,float> &new_filter_coefficients_2D_array,   
				     VectorWithOffset<float> kernel_1d,
				     const float kapa0_over_kapa1)
				     
{

 
  if (kapa0_over_kapa1!=0)
  {

  //kapa0_over_kapa1
  
  // in the case where sq_kappas=1 --- scaled_filter == original template filter 
  Array<2,float> filter_coefficients(IndexRange2D(kernel_1d.get_min_index(), kernel_1d.get_max_index(),
    kernel_1d.get_min_index(), kernel_1d.get_max_index()));
  
  create_kernel_2d (filter_coefficients, kernel_1d);
  
  
#if 1
  // STUFF FOR THE FFT SIZE 
  /****************** *********************************************************************/
  
  const int length_of_size_array = 16;
  const float kapa0_over_kapa1_interval_size=10.F;
  static VectorWithOffset<int> size_for_kapa0_over_kapa1;
  if (size_for_kapa0_over_kapa1.get_length()==0)
  {
    size_for_kapa0_over_kapa1.grow(0,length_of_size_array-1);
    size_for_kapa0_over_kapa1.fill(64);
  }
  
  const int kapa0_over_kapa1_interval = 
    min(static_cast<int>(floor(kapa0_over_kapa1/kapa0_over_kapa1_interval_size)),
    length_of_size_array-1);
  
  float sq_kapas = kapa0_over_kapa1; 
  /******************************************************************************************/
  
  if ( sq_kapas > 10000)
  {
    new_filter_coefficients_2D_array.grow(IndexRange2D(0,0,0,0));
  }
  else if (sq_kapas!=1.F)
  {
    
    while(true)
    {
      const int size = size_for_kapa0_over_kapa1[kapa0_over_kapa1_interval];
      
      int filter_length = static_cast<int>(floor(kernel_1d.get_length()/2));
      
      //cerr << "Now doing size " << size << std::endl;
      
      // FIRST PADD 1D FILTER COEFFICIENTS AND MAKE THEM SYMMETRIC 
      // ( DO NOT TAKE IMAGINARY PART INTO ACCOUNT YET)
      /**********************************************************************************/
      
      Array<1,float> filter_coefficients_padded_1D_array(1,size);
      
      filter_coefficients_padded_1D_array[1] = kernel_1d[0];  
      for ( int i = 1;i<=filter_length;i++)
      {
	filter_coefficients_padded_1D_array[i+1] = kernel_1d[i];    
	filter_coefficients_padded_1D_array[size-(i-1)] = kernel_1d[i];
	
      }

     
      
      /*************************************************************************************/
      
      VectorWithOffset < float> kernel_1d_vector;
      kernel_1d_vector.grow(1,size);
      for ( int i = 1; i<= size ;i++)
	kernel_1d_vector[i] = filter_coefficients_padded_1D_array[i];
      
      Array<2,float> filter_coefficients_padded(IndexRange2D(1,size,1,size));
      
      create_kernel_2d (filter_coefficients_padded, kernel_1d_vector);

  
      
      // rescale to DC=1
      filter_coefficients_padded /= filter_coefficients_padded.sum();
      
      Array<2,float>& fft_filter = filter_coefficients_padded;
      
      float inverse_sq_kapas;
      if (fabs((double)sq_kapas ) >0.000000000001)
	inverse_sq_kapas = 1/sq_kapas;
      else 
	inverse_sq_kapas = 0;
      
      
     // Array<1,float> fft_1_1D_array (1, 2*fft_filter.get_length()*fft_filter[fft_filter.get_min_index()].get_length() *fft_filter[fft_filter.get_min_index()][fft_filter.get_min_index()].get_length());
      Array<1,float> fft_filter_1D_array(1, 2*fft_filter.get_length()*fft_filter[fft_filter.get_min_index()].get_length() );
      
      static shared_ptr <Array<1,float> > fft_filter_1D_array_64 = 
	new Array<1,float>(1,2*64*64);
      static shared_ptr <Array<1,float> > fft_filter_1D_array_128 =
	new  Array<1,float> (1,2*128*128);
      static shared_ptr <Array<1,float> > fft_filter_1D_array_256 = 
	new  Array<1,float> (1,2*256*256);
      static shared_ptr <Array<1,float> > fft_filter_1D_array_512 = 
	new  Array<1,float> (1,2*512*512);
      static shared_ptr <Array<1,float> > fft_filter_1D_array_1024 = 
	new  Array<1,float> (1,2*1024*1024);
      static shared_ptr <Array<1,float> > fft_filter_1D_array_2048= 
	new  Array<1,float> (1,2*2048*2048);
      
      
      convert_array_2D_into_1D_array(fft_filter_1D_array,fft_filter);
     // fft_1_1D_array[1]=1;
      
      Array<1, int> array_lengths(1,2);
      array_lengths[1] = fft_filter.get_length();
      array_lengths[2] = fft_filter[fft_filter.get_min_index()].get_length();
      //array_lengths[3] = fft_filter[fft_filter.get_min_index()][fft_filter.get_min_index()].get_length();
      
      // initialise to 0 to prevent from warnings
      float  fft_1_1D_array  = 0; 
      
     if (size == 64)
      {
	if ( (*fft_filter_1D_array_64)[1] == 0.F)
	{
	  fourn(fft_filter_1D_array, array_lengths, 2,1);
	  fft_filter_1D_array /= sqrt(static_cast<double>(size *size));      
	  *fft_filter_1D_array_64 = fft_filter_1D_array;
	}
	else
	{
	  fft_filter_1D_array = *fft_filter_1D_array_64;

	}
      }
      else if (size ==128)
      {
	if ( (*fft_filter_1D_array_128)[1] == 0.F)
	{
	  fourn(fft_filter_1D_array, array_lengths, 2,1);
          fft_filter_1D_array /= sqrt(static_cast<double>(size *size));      
	  *fft_filter_1D_array_128 = fft_filter_1D_array;
	}
	else
	{
	  fft_filter_1D_array = *fft_filter_1D_array_128;

	}
      }
      else if ( size == 256)
      {
	if ( (*fft_filter_1D_array_256)[1] == 0.F)
	{
	  fourn(fft_filter_1D_array, array_lengths, 2,1);
          fft_filter_1D_array /= sqrt(static_cast<double>(size *size));      
	  *fft_filter_1D_array_256 = fft_filter_1D_array;
	}
	else
	{
	  fft_filter_1D_array = *fft_filter_1D_array_256;

	}
      }
      else if ( size == 512)
      {
	if ( (*fft_filter_1D_array_512)[1] == 0.F)
	{
	  fourn(fft_filter_1D_array, array_lengths, 2,1);
          fft_filter_1D_array /= sqrt(static_cast<double>(size *size));      
	  *fft_filter_1D_array_512 = fft_filter_1D_array;
	}
	else
	{
	  fft_filter_1D_array = *fft_filter_1D_array_512;

	}
      }  
       else if ( size == 1024)
      {
	if ( (*fft_filter_1D_array_1024)[1] == 0.F)
	{
	  fourn(fft_filter_1D_array, array_lengths, 2,1);
          fft_filter_1D_array /= sqrt(static_cast<double>(size *size));      
	  *fft_filter_1D_array_1024 = fft_filter_1D_array;
	}
	else
	{
	  fft_filter_1D_array = *fft_filter_1D_array_1024;

	}
      }  
      else if ( size == 2048)
      {
	if ( (*fft_filter_1D_array_2048)[1] == 0.F)
	{
	  fourn(fft_filter_1D_array, array_lengths, 2,1);
          fft_filter_1D_array /= sqrt(static_cast<double>(size *size));      
	  *fft_filter_1D_array_2048 = fft_filter_1D_array;
	}
	else
	{
	  fft_filter_1D_array = *fft_filter_1D_array_2048;

	}
      }  
      else     
      {
	warning("\nModifiedInverseAveragingImageFilter: Cannot do this at the moment -- size is too big'.\n");
	
      }
      
     // fourn(fft_filter_1D_array, array_lengths, 3,1);
      
      // WARNING  -- this only works for the FFT where the convention is that the final result
      // obtained from the FFT is divided with sqrt(N*N*N)
      switch (size)
      {          
      case 64:
	fft_1_1D_array = static_cast<float>(1/sqrt(static_cast<float>(64*64)));
	break;
      case 128:
	fft_1_1D_array = static_cast<float>(1/sqrt(static_cast<float>(128*128)));
	break;
      case 256:
	fft_1_1D_array = static_cast<float>(1/sqrt(static_cast<float>(256*256)));
	break;
      case 512:
	fft_1_1D_array = static_cast<float>(1/sqrt(static_cast<float>(512*512)));
	break;  
      case 1024:
	fft_1_1D_array = static_cast<float>(1/sqrt(static_cast<float>(1024*1024)));
	break;
      case 2048:
	fft_1_1D_array = static_cast<float>(1/sqrt(static_cast<float>(2048*2048)));
	break;
      
      default:
	warning("\nModifiedInverseAveragingImageFilter: Cannot do this at the moment -- size is too big'.\n");;
	break;
      }
      
      // to check the outputs make the fft consistant with mathematica
      // divide 1/sqrt(size)
      //fft_1_1D_array /= sqrt(static_cast<double> (size *size*size));
     // fft_filter_1D_array /= sqrt(static_cast<double>(size *size*size));
     Array<2,float> new_filter_coefficients_2D_array_tmp (IndexRange2D(1,filter_coefficients_padded.get_length(),1,filter_coefficients_padded.get_length()));
     

      
      {        
	Array<1,float> fft_filter_num_1D_array(1, 2*fft_filter.get_length()*fft_filter[fft_filter.get_min_index()].get_length());
	//Array<1,float> div_1D_array(1, 2*fft_filter.get_length()*fft_filter[fft_filter.get_min_index()].get_length());    
	
	//mulitply_complex_arrays(fft_filter_num_1D_array,fft_filter_1D_array,fft_1_1D_array);
	
	fft_filter_num_1D_array = fft_filter_1D_array* fft_1_1D_array;

	  fft_filter_1D_array *= (sq_kapas-1);
	  // this is necesssery since the imagainary part is stored in the odd indices
	  for ( int i = fft_filter_1D_array.get_min_index(); i<=fft_filter_1D_array.get_max_index(); i+=2)
	  {
	  fft_filter_1D_array[i] += fft_1_1D_array;
	  }
	  fft_filter_1D_array /= sq_kapas;

     	divide_complex_arrays(fft_filter_num_1D_array,fft_filter_1D_array);  


 
	fourn(fft_filter_num_1D_array, array_lengths,2,-1);	
	
	// make it consistent with mathemematica -- the output of the       
	fft_filter_num_1D_array  /= sqrt(static_cast<double>(size *size));
        

   	
	// take the real part only 
	/*********************************************************************************/
	{
	  Array<1,float> real_div_1D_array(1,fft_filter.get_length()*fft_filter[fft_filter.get_min_index()].get_length());
	  
	  for (int i=0;i<=(size*size)-1;i++)
	    real_div_1D_array[i+1] = fft_filter_num_1D_array[2*i+1];
	  
	  /*********************************************************************************/
  
	  convert_array_1D_into_2D_array(new_filter_coefficients_2D_array_tmp,real_div_1D_array);
 	  
	}
      }


	  int kernel_length_x=0;
	  int kernel_length_y=0;
	  
	  // to prevent form aliasing limit the new range for the coefficients to 
	  // filter_coefficients_padded.get_length()/4
	  
	  //cerr << " X DIERCTION NOW" << endl;
	  // do the x -direction first -- fix y and z to the min and look for the max index in x
	  int jy = new_filter_coefficients_2D_array_tmp.get_min_index();
	  for (int i=new_filter_coefficients_2D_array_tmp[jy].get_min_index();i<=filter_coefficients_padded[jy].get_max_index()/2;i++)
	  {
	    if (fabs((double)new_filter_coefficients_2D_array_tmp[jy][i])
	      <= new_filter_coefficients_2D_array_tmp[new_filter_coefficients_2D_array_tmp.get_min_index()][new_filter_coefficients_2D_array_tmp.get_min_index()]*1/1000000) break;
	    else (kernel_length_x)++;
	  }
	  
	  /******************************* Y DIRECTION ************************************/
	  
	  
	  int ix = new_filter_coefficients_2D_array_tmp[new_filter_coefficients_2D_array_tmp.get_min_index()].get_min_index();
	  for (int j=new_filter_coefficients_2D_array_tmp.get_min_index();j<=filter_coefficients_padded.get_max_index()/2;j++)
	  {
	    if (fabs((double)new_filter_coefficients_2D_array_tmp[j][ix])
	      //= new_filter_coefficients_3D_array_tmp[new_filter_coefficients_3D_array_tmp.get_min_index()][new_filter_coefficients_3D_array_tmp.get_min_index()][new_filter_coefficients_3D_array_tmp.get_min_index()]*1/100000000) break;
	      <= new_filter_coefficients_2D_array_tmp[new_filter_coefficients_2D_array_tmp.get_min_index()][new_filter_coefficients_2D_array_tmp.get_min_index()]*1/1000000) break;
	    else (kernel_length_y)++;
	  }

	  
	  /********************************************************************************/
	  

	  /********************************************************************************/
	  
	  if (kernel_length_x == filter_coefficients_padded.get_length()/4)
	  {
	    warning("ModifiedInverseAverigingArrayFilter3D: kernel_length_x reached maximum length %d. "
	      "First filter coefficient %g, last %g, kappa0_over_kappa1 was %g\n"
	      "Increasing length of FFT array to resolve this problem\n",
	      kernel_length_x, new_filter_coefficients_2D_array_tmp[new_filter_coefficients_2D_array_tmp.get_min_index()][new_filter_coefficients_2D_array_tmp.get_min_index()],
	      new_filter_coefficients_2D_array_tmp[new_filter_coefficients_2D_array_tmp.get_min_index()][kernel_length_x],
	      kapa0_over_kapa1);
	    size_for_kapa0_over_kapa1[kapa0_over_kapa1_interval]*=2;
	    for (int i=kapa0_over_kapa1_interval+1; i<size_for_kapa0_over_kapa1.get_length(); ++i)
	      size_for_kapa0_over_kapa1[i]=
	      max(size_for_kapa0_over_kapa1[i], size_for_kapa0_over_kapa1[kapa0_over_kapa1_interval]);
	  }
	  else if (kernel_length_y == filter_coefficients_padded.get_length()/4)
	  {
	    warning("ModifiedInverseAverigingArrayFilter3D: kernel_length_y reached maximum length %d. "
	      "First filter coefficient %g, last %g, kappa0_over_kappa1 was %g\n"
	      "Increasing length of FFT array to resolve this problem\n",
	      kernel_length_y, new_filter_coefficients_2D_array_tmp[new_filter_coefficients_2D_array_tmp.get_min_index()][new_filter_coefficients_2D_array_tmp.get_min_index()], new_filter_coefficients_2D_array_tmp[kernel_length_y][new_filter_coefficients_2D_array_tmp.get_min_index()],
	      kapa0_over_kapa1);
	    size_for_kapa0_over_kapa1[kapa0_over_kapa1_interval]*=2;
	    for (int i=kapa0_over_kapa1_interval+1; i<size_for_kapa0_over_kapa1.get_length(); ++i)
	      size_for_kapa0_over_kapa1[i]=
	      max(size_for_kapa0_over_kapa1[i], size_for_kapa0_over_kapa1[kapa0_over_kapa1_interval]);
	  }
	  else
	  {
	
/*	 cerr << " calulcated coefficients " << endl;
	 for (int j=1;j<=4;j++)	  
	  for (int i=1;i<=4;i++)
	   {
	     cerr << new_filter_coefficients_2D_array_tmp[j][i] << "   " ;

	   }	  cerr << endl;*/
	    	
	    new_filter_coefficients_2D_array.grow(IndexRange2D(
	      -(kernel_length_y-1),kernel_length_y-1,
	      -(kernel_length_x-1),kernel_length_x-1));
	    
	    new_filter_coefficients_2D_array[0][0] = new_filter_coefficients_2D_array_tmp[1][1];

	   // new_filter_coefficients_2D_array[kernel_length_y-1][kernel_length_x-1] = new_filter_coefficients_2D_array_tmp[kernel_length_y][kernel_length_x];
	    	    
	      for (int  j = 0;j<= kernel_length_y-1;j++)	  
		for (int  i = 0;i<= kernel_length_x-1;i++)	  		  
		{
		  new_filter_coefficients_2D_array[j][i]=new_filter_coefficients_2D_array_tmp[j+1][i+1];
		  new_filter_coefficients_2D_array[-j][-i]=new_filter_coefficients_2D_array_tmp[j+1][i+1];

		  new_filter_coefficients_2D_array[-j][i]=new_filter_coefficients_2D_array_tmp[j+1][i+1];
		  new_filter_coefficients_2D_array[j][-i]=new_filter_coefficients_2D_array_tmp[j+1][i+1];


		  
		}
        

      break; // out of while(true)
	  }
    } // this bracket is for the while loop
    }
    else //sq_kappas == 1
    {
      new_filter_coefficients_2D_array = filter_coefficients;
    }
    
    
#endif 
    
           
    // rescale to DC=1
    float sum_new_coefficients =0;  
      for (int j=new_filter_coefficients_2D_array.get_min_index();j<=new_filter_coefficients_2D_array.get_max_index();j++)	  
	for (int i=new_filter_coefficients_2D_array[j].get_min_index();i<=new_filter_coefficients_2D_array[j].get_max_index();i++)
	  sum_new_coefficients += new_filter_coefficients_2D_array[j][i];  
	
	  for (int j=new_filter_coefficients_2D_array.get_min_index();j<=new_filter_coefficients_2D_array.get_max_index();j++)	  
	    for (int i=new_filter_coefficients_2D_array[j].get_min_index();i<=new_filter_coefficients_2D_array[j].get_max_index();i++)
	      new_filter_coefficients_2D_array[j][i] /=sum_new_coefficients;  

    /*	cerr << " now assigned symmetric coeff " << endl;
	   for (int  j = new_filter_coefficients_2D_array.get_min_index();j<= new_filter_coefficients_2D_array.get_max_index();j++)	  
		for (int  i = new_filter_coefficients_2D_array[j].get_min_index();i<=new_filter_coefficients_2D_array[j].get_max_index();i++)
	   {
	     cerr << new_filter_coefficients_2D_array[j][i] << "   " ;

	   }*/
    
	
}
else
{
   new_filter_coefficients_2D_array.grow(IndexRange2D(0,0,0,0));
   new_filter_coefficients_2D_array[0][0] =0;
  }
		    
}



#if 1


template <typename elemT>
ModifiedInverseAveragingImageFilterAll<elemT>::
ModifiedInverseAveragingImageFilterAll()
{ 
  set_defaults();
}


template <typename elemT>
ModifiedInverseAveragingImageFilterAll<elemT>::
ModifiedInverseAveragingImageFilterAll(string proj_data_filename_v,
				    string attenuation_proj_data_filename_v,
				    const VectorWithOffset<elemT>& filter_coefficients_v,
				    shared_ptr<ProjData> proj_data_ptr_v,
				    shared_ptr<ProjData> attenuation_proj_data_ptr_v,
				    DiscretisedDensity<3,float>* initial_image_v,
				    DiscretisedDensity<3,float>* sensitivity_image_v,
				    DiscretisedDensity<3,float>* precomputed_coefficients_image_v,
				    DiscretisedDensity<3,float>* normalised_bck_image_v,
				    int mask_size_v,  int num_dim_v)

				    
{
  assert(filter_coefficients.get_length() == 0 ||
         filter_coefficients.begin()==0);
  
  for (int i = filter_coefficients_v.get_min_index();i<=filter_coefficients_v.get_max_index();i++)
    filter_coefficients[i] = filter_coefficients_v[i];
  proj_data_filename  = proj_data_filename_v;
  attenuation_proj_data_filename = attenuation_proj_data_filename_v;
  proj_data_ptr = proj_data_ptr_v;
  attenuation_proj_data_ptr = attenuation_proj_data_ptr_v;
  initial_image = initial_image_v;
  sensitivity_image = sensitivity_image_v;
  precomputed_coefficients_image = precomputed_coefficients_image_v;
  normalised_bck_image = normalised_bck_image_v;
  mask_size= mask_size_v;
  num_dim = num_dim_v;
}


template <typename elemT>
Succeeded 
ModifiedInverseAveragingImageFilterAll<elemT>::
virtual_set_up(const DiscretisedDensity<3,elemT>& density)
{
    proj_data_ptr = 
       ProjData::read_from_file( proj_data_filename); 
    
    if (attenuation_proj_data_filename !="1")
    attenuation_proj_data_ptr =
    ProjData::read_from_file(attenuation_proj_data_filename); 
	else 
    attenuation_proj_data_ptr = NULL;

   if (initial_image_filename !="1")
    initial_image =
    DiscretisedDensity<3,float>::read_from_file(initial_image_filename); 
	else 
    initial_image  = NULL;

   if (sensitivity_image_filename !="1")
    sensitivity_image =
    DiscretisedDensity<3,float>::read_from_file(sensitivity_image_filename); 
	else 
    sensitivity_image = NULL;
   if (precomputed_coefficients_filename !="1")
     precomputed_coefficients_image = 
      DiscretisedDensity<3,float>::read_from_file(precomputed_coefficients_filename);
   else
     precomputed_coefficients_image =NULL; 

   if (normalised_bck_filename !="1")
     normalised_bck_image = 
      DiscretisedDensity<3,float>::read_from_file(normalised_bck_filename);
   else
     normalised_bck_image =NULL; 



    return Succeeded::yes;
  
}


template <typename elemT>
void 
ModifiedInverseAveragingImageFilterAll<elemT>::precalculate_filter_coefficients (VectorWithOffset < VectorWithOffset < VectorWithOffset <shared_ptr <ArrayFilter3DUsingConvolution <float> >  > > >& all_filter_coefficients,
									      DiscretisedDensity<3,elemT>* in_density) const
{

 
  VectorWithOffset < shared_ptr <ArrayFilter3DUsingConvolution <float> > > filter_lookup;
  filter_lookup.grow(1,500);
  const int k_min =1;
  const float k_interval = 0.01F; //0.01F;
  

  shared_ptr<ProjDataInfo> new_data_info_ptr  = proj_data_ptr->get_proj_data_info_ptr()->clone();
  VoxelsOnCartesianGrid<float>* in_density_cast =
    dynamic_cast< VoxelsOnCartesianGrid<float>* >(in_density); 
  

  VoxelsOnCartesianGrid<float> *  vox_image_ptr_1 =
    new VoxelsOnCartesianGrid<float> (IndexRange3D(in_density_cast->get_min_z(),in_density_cast->get_max_z(),
    in_density_cast->get_min_y(),in_density_cast->get_max_y(),
    in_density_cast->get_min_x(),in_density_cast->get_max_x()),
    in_density_cast->get_origin(),in_density_cast->get_voxel_size());  
  
  int start_segment_num = proj_data_ptr->get_min_segment_num();
  int end_segment_num = proj_data_ptr->get_max_segment_num();
  
  VectorWithOffset<SegmentByView<float> *> all_segments(start_segment_num, end_segment_num);
  VectorWithOffset<SegmentByView<float> *> all_segments_for_kappa0(start_segment_num, end_segment_num);   
  VectorWithOffset<SegmentByView<float> *> all_attenuation_segments(start_segment_num, end_segment_num);
  
  
  // first initialise to false
  bool do_attenuation = false;

  for (int segment_num = start_segment_num; segment_num <= end_segment_num; ++segment_num)
  {
    all_segments[segment_num] = new SegmentByView<float>(proj_data_ptr->get_empty_segment_by_view(segment_num));
    all_segments_for_kappa0[segment_num] = new SegmentByView<float>(proj_data_ptr->get_empty_segment_by_view(segment_num));
    
    if (attenuation_proj_data_filename!="1")
    {
      do_attenuation = true;
      all_attenuation_segments[segment_num] = 
	new SegmentByView<float>(attenuation_proj_data_ptr->get_segment_by_view(segment_num));
    }
    else 
    {
      do_attenuation = false;
      all_attenuation_segments[segment_num] = new SegmentByView<float>(proj_data_ptr->get_empty_segment_by_view(segment_num));		 
      (*all_attenuation_segments[segment_num]).fill(1);
    }
    
  }
  
  vox_image_ptr_1->set_origin(Coordinate3D<float>(0,0,0));   
  
  shared_ptr<DiscretisedDensity<3,float> > image_sptr =  vox_image_ptr_1;
  
  shared_ptr<ProjMatrixByDensel> proj_matrix_ptr = 
    new ProjMatrixByDenselUsingRayTracing;
  
  proj_matrix_ptr->set_up(proj_data_ptr->get_proj_data_info_ptr()->clone(),
    image_sptr);
  cerr << proj_matrix_ptr->parameter_info();
  
  fwd_densels_all(all_segments,proj_matrix_ptr, proj_data_ptr,
    in_density_cast->get_min_z(), in_density_cast->get_max_z(),
    in_density_cast->get_min_y(), in_density_cast->get_max_y(),
    in_density_cast->get_min_x(), in_density_cast->get_max_x(),
    *in_density);
  
  VoxelsOnCartesianGrid<float> *  vox_image_ptr_kappa0 =
    new VoxelsOnCartesianGrid<float>(IndexRange3D(in_density_cast->get_min_z(),in_density_cast->get_max_z(),
    in_density_cast->get_min_y(),in_density_cast->get_max_y(),
    in_density_cast->get_min_x(),in_density_cast->get_max_x()),
    in_density_cast->get_origin(),in_density_cast->get_voxel_size());  
  
  VoxelsOnCartesianGrid<float> *  vox_image_ptr_kappa1 =
    new VoxelsOnCartesianGrid<float>(IndexRange3D(in_density_cast->get_min_z(),in_density_cast->get_max_z(),
    in_density_cast->get_min_y(),in_density_cast->get_max_y(),
    in_density_cast->get_min_x(),in_density_cast->get_max_x()),
    in_density_cast->get_origin(),in_density_cast->get_voxel_size());  

   VoxelsOnCartesianGrid<float> *  kappa_coefficients =
    new VoxelsOnCartesianGrid<float>(IndexRange3D(in_density_cast->get_min_z(),in_density_cast->get_max_z(),
    in_density_cast->get_min_y(),in_density_cast->get_max_y(),
    in_density_cast->get_min_x(),in_density_cast->get_max_x()),
    in_density_cast->get_origin(),in_density_cast->get_voxel_size());
  
  
  shared_ptr<DiscretisedDensity<3,float> > kappa0_ptr_bck =  vox_image_ptr_kappa0;       
  shared_ptr<DiscretisedDensity<3,float> > kappa1_ptr_bck =  vox_image_ptr_kappa1;   
  
  // WARNING - find a way of finding max in the sinogram
  // TODO - include other segments as well
  float max_in_viewgram =0.F;
  
  for (int segment_num = start_segment_num; segment_num<= end_segment_num; segment_num++) 
  {
    SegmentByView<float> segment_by_view = 
      proj_data_ptr->get_segment_by_view(segment_num);
    const float current_max_in_viewgram = segment_by_view.find_max();
    if ( current_max_in_viewgram >= max_in_viewgram)
      max_in_viewgram = current_max_in_viewgram ;
    else
      continue;
  }
  const float threshold = 0.0001F*max_in_viewgram;  
  
  cerr << " THRESHOLD IS" << threshold; 
  cerr << endl;
  
  find_inverse_and_bck_densels(*kappa1_ptr_bck,all_segments,
    all_attenuation_segments,
    vox_image_ptr_kappa1->get_min_z(),vox_image_ptr_kappa1->get_max_z(),
    vox_image_ptr_kappa1->get_min_y(),vox_image_ptr_kappa1->get_max_y(),
    vox_image_ptr_kappa1->get_min_x(),vox_image_ptr_kappa1->get_max_x(),
    *proj_matrix_ptr, do_attenuation,threshold, false); //true);
  
  for (int segment_num = start_segment_num; segment_num <= end_segment_num; ++segment_num)
  { 
    delete all_segments[segment_num];
    delete all_attenuation_segments[segment_num];
  }   
  
  cerr << "min and max in image - kappa1 " <<kappa1_ptr_bck->find_min()
    << ", " << kappa1_ptr_bck->find_max() << endl;   
  
  for (int k=in_density_cast->get_min_z();k<=in_density_cast->get_max_z();k++)   
    for (int j =in_density_cast->get_min_y();j<=in_density_cast->get_max_y();j++)
      for (int i =in_density_cast->get_min_x();i<=in_density_cast->get_max_x();i++)	
      {
	
	// WARNING - only works for segment zero at the moment
	// do the calculation of kappa0 here
	kappa0_ptr_bck->fill(0); 
	for (int segment_num = start_segment_num; 
	segment_num <= end_segment_num; ++segment_num)
	{  
	  (*all_segments_for_kappa0[segment_num]).fill(0);	    
	}
	if (true) //attenuation_proj_data_filename !="1")
	{
	   shared_ptr< VoxelsOnCartesianGrid<float> > in_density_cast_tmp =
	   new VoxelsOnCartesianGrid<float>
	    (IndexRange3D(-mask_size+(ceil(in_density_cast->get_max_z()-in_density_cast->get_min_z())/2),
	     mask_size+(ceil(in_density_cast->get_max_z()-in_density_cast->get_min_z())/2),
	    -mask_size+6,mask_size+6,
	    -mask_size+6,mask_size+6),in_density_cast->get_origin(),in_density_cast->get_voxel_size());  
	 /*shared_ptr< VoxelsOnCartesianGrid<float> > in_density_cast_tmp =
	    new VoxelsOnCartesianGrid<float>(IndexRange3D(k,k,
	      //-mask_size+k,mask_size+k,
	    -mask_size+6,mask_size+6,
	    -mask_size+6,mask_size+6),in_density_cast->get_origin(),in_density_cast->get_voxel_size());  */
	  CPUTimer timer;
	  timer.start();
	  
	  	// SM 23/05/2002 mask now 3D
	  const int min_k = max(in_density_cast->get_min_z(),k-mask_size);
	  const int max_k = min(in_density_cast->get_max_z(),k+mask_size);			
	  const int min_j = max(in_density_cast->get_min_y(),j-mask_size);
	  const int max_j = min(in_density_cast->get_max_y(),j+mask_size);
	  const int min_i = max(in_density_cast->get_min_x(),i-mask_size);
	  const int max_i = min(in_density_cast->get_max_x(),i+mask_size);
	  
	  // the mask size is in 2D only
	  // SM mask now 3D 
	 for (int k_in =min_k;k_in<=max_k;k_in++)
	  for (int j_in =min_j;j_in<=max_j;j_in++)
	    for (int i_in =min_i;i_in<=max_i;i_in++)	
	    {
	      (*in_density_cast_tmp)[k_in-k+(ceil(in_density_cast->get_max_z()-in_density_cast->get_min_z())/2)][j_in-j +6][i_in-i+6] = (*in_density_cast)[k_in][j_in][i_in];
	      //(*in_density_cast_tmp)[k][j_in-j +6][i_in-i+6] = (*in_density_cast)[k][j_in][i_in];
	    }
	    
	    fwd_densels_all(all_segments_for_kappa0,proj_matrix_ptr, proj_data_ptr,
	      in_density_cast_tmp->get_min_z(), in_density_cast_tmp->get_max_z(),
	      in_density_cast_tmp->get_min_y(),in_density_cast_tmp->get_max_y(),
	      in_density_cast_tmp->get_min_x(),in_density_cast_tmp->get_max_x(),
	      *in_density_cast_tmp);
	    
	    find_inverse_and_bck_densels(*kappa0_ptr_bck,all_segments_for_kappa0,
	      all_attenuation_segments,
	     //k,k,
	      (ceil(vox_image_ptr_kappa1->get_max_z()-vox_image_ptr_kappa1->get_min_z()))/2,ceil((vox_image_ptr_kappa1->get_max_z()-vox_image_ptr_kappa1->get_min_z())/2),
	      //0,0,0,0,
	      6,6,6,6,
	      *proj_matrix_ptr,false,threshold, false) ;//true);	  
	    (*kappa0_ptr_bck)[k][j][i] = (*kappa0_ptr_bck)[(ceil(vox_image_ptr_kappa1->get_max_z()-vox_image_ptr_kappa1->get_min_z()))/2][6][6];
	    //(*kappa0_ptr_bck)[k][j][i] = (*kappa0_ptr_bck)[k][6][6];
	    
	    timer.stop();
	    //cerr << "kappa0 time "<< timer.value() << endl;
	}
	else
	{
	  const int min_j = max(in_density_cast->get_min_y(),j-mask_size);
	  const int max_j = min(in_density_cast->get_max_y(),j+mask_size);
	  const int min_i = max(in_density_cast->get_min_x(),i-mask_size);
	  const int max_i = min(in_density_cast->get_max_x(),i+mask_size);
	  
	  fwd_densels_all(all_segments_for_kappa0,proj_matrix_ptr, proj_data_ptr,
	    in_density_cast->get_min_z(), in_density_cast->get_max_z(),
	    min_j,max_j,
	    min_i,max_i,
	    //j-2,j+2,
	    //i-2,i+2,
	    *in_density_cast);
	  
	  find_inverse_and_bck_densels(*kappa0_ptr_bck,all_segments_for_kappa0,
	    all_attenuation_segments,
	    vox_image_ptr_kappa1->get_min_z(),vox_image_ptr_kappa1->get_max_z(),
	    j,j,i,i,
	    *proj_matrix_ptr,false,threshold, true);
	  
	}
	float sq_kapas;
//	float multiply_with_sensitivity;
	if ( fabs((double)(*kappa1_ptr_bck)[k][j][i]) > 0.00000000000001 && 
	  fabs((double)(*kappa0_ptr_bck)[k][j][i]) > 0.00000000000001 )
	{ 
	  sq_kapas =((*kappa0_ptr_bck)[k][j][i]*(*kappa0_ptr_bck)[k][j][i])/((*kappa1_ptr_bck)[k][j][i]*(*kappa1_ptr_bck)[k][j][i]);
	  // cerr << "sq_kapas "  << sq_kapas << endl;

	  //float tmp = (*sensitivity_image)[k][j][i];
	  //float tmp_1 = (*sensitivity_image)[ceil(in_density_cast->get_max_z()-in_density_cast->get_min_z())/2][6][6];
	    //cerr << (*sensitivity_image)[k][j][i] << "   " << (*sensitivity_image)[ceil(in_density_cast->get_max_z()-in_density_cast->get_min_z())/2][6][6];

	 /* multiply_with_sensitivity = (*sensitivity_image)[ceil(in_density_cast->get_max_z()-in_density_cast->get_min_z())/2][6][6];
	  if ((*sensitivity_image)[k][j][i] >0.0000001 || (*sensitivity_image)[k][j][i] <-0.0000001)
	  {
	  multiply_with_sensitivity /= (*sensitivity_image)[k][j][i];
	  }
	  else
	  {
	    multiply_with_sensitivity /= 0.000001F;
	  }*/
	  
	 // sq_kapas *= multiply_with_sensitivity;
	  (*kappa_coefficients)[k][j][i] = sq_kapas;

	  //sq_kapas = 10.0F;
	  //cerr << sq_kapas << "   " ;

	  int k_index ;
	  //int  k_index = min(round(((float)sq_kapas- k_min)/k_interval), 1000);
	//  if (sq_kapas > 1000) 
	  //{
	    //k_index = 1000;
	  //}
	  //else
	  //{
	    k_index = round(((float)sq_kapas- k_min)/k_interval);
	  //}
	   if (k_index < 1) 
	   {k_index = 1;}
	    
	   if ( k_index > 500)
	   { k_index  = 500;}
	  
	  
	  if ( filter_lookup[k_index]==NULL )
	  {
	    Array <3,float> new_coeffs;
	    cerr << "\ncomputing new filter for sq_kappas " << sq_kapas << " at index " <<k_index<< std::endl;
	  construct_scaled_filter_coefficients_3D(new_coeffs, filter_coefficients,sq_kapas);
		  filter_lookup[k_index] = new ArrayFilter3DUsingConvolution<float>(new_coeffs);
		  all_filter_coefficients[k][j][i] = filter_lookup[k_index];
		  //new ArrayFilter3DUsingConvolution<float>(new_coeffs);	
		  
	  }
	  else 
	  {
	    all_filter_coefficients[k][j][i] = filter_lookup[k_index];
		  
	  }
		  
	
	//  all_filter_coefficients[k][j][i] = 
	  //  new ModifiedInverseAverigingArrayFilter<3,float>(inverse_filter);		
	  
	}
	else
	{	
	  sq_kapas = 0;
	//  inverse_filter = 
	  //  ModifiedInverseAverigingArrayFilter<3,float>();	  
	  all_filter_coefficients[k][j][i] =
	    new  ArrayFilter3DUsingConvolution<float>();
	  
	}
	
      }   

      
  write_basic_interfile("kappa_coefficients_2D_SENS",*kappa_coefficients);
  delete kappa_coefficients ;

      
  for (int segment_num = start_segment_num; segment_num <= end_segment_num; ++segment_num)
  {
    delete  all_segments_for_kappa0[segment_num];
  }
    
      
      
}


template <typename elemT>
void 
ModifiedInverseAveragingImageFilterAll<elemT>::precalculate_filter_coefficients_2D (VectorWithOffset < VectorWithOffset < VectorWithOffset <shared_ptr <ArrayFilter2DUsingConvolution <float> >  > > >& all_filter_coefficients,
									      DiscretisedDensity<3,elemT>* in_density) const
{
 
  VectorWithOffset < shared_ptr <ArrayFilter2DUsingConvolution <float> > > filter_lookup;
  const int num_elements_in_interval = 500;
  filter_lookup.grow(1,num_elements_in_interval);
  const int k_min =1;
  const float k_interval = 0.01F; //0.01F;
  

  shared_ptr<ProjDataInfo> new_data_info_ptr  = proj_data_ptr->get_proj_data_info_ptr()->clone();
  VoxelsOnCartesianGrid<float>* in_density_cast =
    dynamic_cast< VoxelsOnCartesianGrid<float>* >(in_density); 
  

  VoxelsOnCartesianGrid<float> *  vox_image_ptr_1 =
    new VoxelsOnCartesianGrid<float> (IndexRange3D(in_density_cast->get_min_z(),in_density_cast->get_max_z(),
    in_density_cast->get_min_y(),in_density_cast->get_max_y(),
    in_density_cast->get_min_x(),in_density_cast->get_max_x()),
    in_density_cast->get_origin(),in_density_cast->get_voxel_size());  
  
  int start_segment_num = proj_data_ptr->get_min_segment_num();
  int end_segment_num = proj_data_ptr->get_max_segment_num();
  
  VectorWithOffset<SegmentByView<float> *> all_segments(start_segment_num, end_segment_num);
  VectorWithOffset<SegmentByView<float> *> all_segments_for_kappa0(start_segment_num, end_segment_num);   
  VectorWithOffset<SegmentByView<float> *> all_attenuation_segments(start_segment_num, end_segment_num);
  
  
  // first initialise to false
  bool do_attenuation = false;

  for (int segment_num = start_segment_num; segment_num <= end_segment_num; ++segment_num)
  {
    all_segments[segment_num] = new SegmentByView<float>(proj_data_ptr->get_empty_segment_by_view(segment_num));
    all_segments_for_kappa0[segment_num] = new SegmentByView<float>(proj_data_ptr->get_empty_segment_by_view(segment_num));
    
    if (attenuation_proj_data_filename!="1")
    {
      do_attenuation = true;
      all_attenuation_segments[segment_num] = 
	new SegmentByView<float>(attenuation_proj_data_ptr->get_segment_by_view(segment_num));
    }
    else 
    {
      do_attenuation = false;
      all_attenuation_segments[segment_num] = new SegmentByView<float>(proj_data_ptr->get_empty_segment_by_view(segment_num));		 
      (*all_attenuation_segments[segment_num]).fill(1);
    }
    
  }
  
  vox_image_ptr_1->set_origin(Coordinate3D<float>(0,0,0));   
  
  shared_ptr<DiscretisedDensity<3,float> > image_sptr =  vox_image_ptr_1;
  
  shared_ptr<ProjMatrixByDensel> proj_matrix_ptr = 
    new ProjMatrixByDenselUsingRayTracing;
  
  proj_matrix_ptr->set_up(proj_data_ptr->get_proj_data_info_ptr()->clone(),
    image_sptr);
  cerr << proj_matrix_ptr->parameter_info();
  
  fwd_densels_all(all_segments,proj_matrix_ptr, proj_data_ptr,
    in_density_cast->get_min_z(), in_density_cast->get_max_z(),
    in_density_cast->get_min_y(), in_density_cast->get_max_y(),
    in_density_cast->get_min_x(), in_density_cast->get_max_x(),
    *in_density);
  
  VoxelsOnCartesianGrid<float> *  vox_image_ptr_kappa0 =
    new VoxelsOnCartesianGrid<float>(IndexRange3D(in_density_cast->get_min_z(),in_density_cast->get_max_z(),
    in_density_cast->get_min_y(),in_density_cast->get_max_y(),
    in_density_cast->get_min_x(),in_density_cast->get_max_x()),
    in_density_cast->get_origin(),in_density_cast->get_voxel_size());  
  
  VoxelsOnCartesianGrid<float> *  vox_image_ptr_kappa1 =
    new VoxelsOnCartesianGrid<float>(IndexRange3D(in_density_cast->get_min_z(),in_density_cast->get_max_z(),
    in_density_cast->get_min_y(),in_density_cast->get_max_y(),
    in_density_cast->get_min_x(),in_density_cast->get_max_x()),
    in_density_cast->get_origin(),in_density_cast->get_voxel_size());  

   VoxelsOnCartesianGrid<float> *  kappa_coefficients =
    new VoxelsOnCartesianGrid<float>(IndexRange3D(in_density_cast->get_min_z(),in_density_cast->get_max_z(),
    in_density_cast->get_min_y(),in_density_cast->get_max_y(),
    in_density_cast->get_min_x(),in_density_cast->get_max_x()),
    in_density_cast->get_origin(),in_density_cast->get_voxel_size());
  
  
  shared_ptr<DiscretisedDensity<3,float> > kappa0_ptr_bck =  vox_image_ptr_kappa0;       
  shared_ptr<DiscretisedDensity<3,float> > kappa1_ptr_bck =  vox_image_ptr_kappa1;   
  
  // WARNING - find a way of finding max in the sinogram
  // TODO - include other segments as well
  float max_in_viewgram =0.F;
  
  for (int segment_num = start_segment_num; segment_num<= end_segment_num; segment_num++) 
  {
    SegmentByView<float> segment_by_view = 
      proj_data_ptr->get_segment_by_view(segment_num);
    const float current_max_in_viewgram = segment_by_view.find_max();
    if ( current_max_in_viewgram >= max_in_viewgram)
      max_in_viewgram = current_max_in_viewgram ;
    else
      continue;
  }
  const float threshold = 0.0001F*max_in_viewgram;  
  
  cerr << " THRESHOLD IS" << threshold; 
  cerr << endl;
  
  find_inverse_and_bck_densels(*kappa1_ptr_bck,all_segments,
    all_attenuation_segments,
    vox_image_ptr_kappa1->get_min_z(),vox_image_ptr_kappa1->get_max_z(),
    vox_image_ptr_kappa1->get_min_y(),vox_image_ptr_kappa1->get_max_y(),
    vox_image_ptr_kappa1->get_min_x(),vox_image_ptr_kappa1->get_max_x(),
    *proj_matrix_ptr, do_attenuation,threshold, false); //true);
  
  for (int segment_num = start_segment_num; segment_num <= end_segment_num; ++segment_num)
  { 
    delete all_segments[segment_num];
    delete all_attenuation_segments[segment_num];
  }   
  
  cerr << "min and max in image - kappa1 " <<kappa1_ptr_bck->find_min()
    << ", " << kappa1_ptr_bck->find_max() << endl;   
  
  for (int k=in_density_cast->get_min_z();k<=in_density_cast->get_max_z();k++)   
    for (int j =in_density_cast->get_min_y();j<=in_density_cast->get_max_y();j++)
      for (int i =in_density_cast->get_min_x();i<=in_density_cast->get_max_x();i++)	
      {
	
	// WARNING - only works for segment zero at the moment
	// do the calculation of kappa0 here
	kappa0_ptr_bck->fill(0); 
	for (int segment_num = start_segment_num; 
	segment_num <= end_segment_num; ++segment_num)
	{  
	  (*all_segments_for_kappa0[segment_num]).fill(0);	    
	}
	if (true) //attenuation_proj_data_filename !="1")
	{

#if 1
	   shared_ptr< VoxelsOnCartesianGrid<float> > in_density_cast_tmp =
	   new VoxelsOnCartesianGrid<float>
	    (IndexRange3D(k,k,
	    //-mask_size+(ceil(in_density_cast->get_max_z()-in_density_cast->get_min_z())/2),
	    // mask_size+(ceil(in_density_cast->get_max_z()-in_density_cast->get_min_z())/2),
	    -mask_size+6,mask_size+6,
	    -mask_size+6,mask_size+6),in_density_cast->get_origin(),in_density_cast->get_voxel_size());  
	 /*shared_ptr< VoxelsOnCartesianGrid<float> > in_density_cast_tmp =
	    new VoxelsOnCartesianGrid<float>(IndexRange3D(k,k,
	      //-mask_size+k,mask_size+k,
	    -mask_size+6,mask_size+6,
	    -mask_size+6,mask_size+6),in_density_cast->get_origin(),in_density_cast->get_voxel_size());  */
	  CPUTimer timer;
	  timer.start();
	  
	  	// SM 23/05/2002 mask now 3D
	  //const int min_k = in_density_cast->get_min_z(); //max(in_density_cast->get_min_z(),k-mask_size);
	  //const int max_k = in_density_cast->get_max_z(); // min(in_density_cast->get_max_z(),k+mask_size);			
	  const int min_j = max(in_density_cast->get_min_y(),j-mask_size);
	  const int max_j = min(in_density_cast->get_max_y(),j+mask_size);
	  const int min_i = max(in_density_cast->get_min_x(),i-mask_size);
	  const int max_i = min(in_density_cast->get_max_x(),i+mask_size);
	  
	  // the mask size is in 2D only
	  // SM mask now 3D 
	 //for (int k_in =min_k;k_in<=max_k;k_in++)
	  for (int j_in =min_j;j_in<=max_j;j_in++)
	    for (int i_in =min_i;i_in<=max_i;i_in++)	
	    {
	      (*in_density_cast_tmp)[k][j_in-j +6][i_in-i+6] = (*in_density_cast)[k][j_in][i_in];
	      //(*in_density_cast_tmp)[k][j_in-j +6][i_in-i+6] = (*in_density_cast)[k][j_in][i_in];
	    }
	    
	    fwd_densels_all(all_segments_for_kappa0,proj_matrix_ptr, proj_data_ptr,
	      in_density_cast_tmp->get_min_z(), in_density_cast_tmp->get_max_z(),
	      in_density_cast_tmp->get_min_y(),in_density_cast_tmp->get_max_y(),
	      in_density_cast_tmp->get_min_x(),in_density_cast_tmp->get_max_x(),
	      *in_density_cast_tmp);
	    
	    find_inverse_and_bck_densels(*kappa0_ptr_bck,all_segments_for_kappa0,
	      all_attenuation_segments,
	     k,k,
	      //(ceil(vox_image_ptr_kappa1->get_max_z()-vox_image_ptr_kappa1->get_min_z()))/2,ceil((vox_image_ptr_kappa1->get_max_z()-vox_image_ptr_kappa1->get_min_z())/2),
	      //0,0,0,0,
	      6,6,6,6,
	      *proj_matrix_ptr,false,threshold, false) ;//true);	  
	    //(*kappa0_ptr_bck)[k][j][i] = (*kappa0_ptr_bck)[(ceil(vox_image_ptr_kappa1->get_max_z()-vox_image_ptr_kappa1->get_min_z()))/2][6][6];
	    (*kappa0_ptr_bck)[k][j][i] = (*kappa0_ptr_bck)[k][6][6];
	    
	    timer.stop();
	    //cerr << "kappa0 time "<< timer.value() << endl;
	}
	else
	{
	  const int min_j = max(in_density_cast->get_min_y(),j-mask_size);
	  const int max_j = min(in_density_cast->get_max_y(),j+mask_size);
	  const int min_i = max(in_density_cast->get_min_x(),i-mask_size);
	  const int max_i = min(in_density_cast->get_max_x(),i+mask_size);
	  
	  fwd_densels_all(all_segments_for_kappa0,proj_matrix_ptr, proj_data_ptr,
	    in_density_cast->get_min_z(), in_density_cast->get_max_z(),
	    min_j,max_j,
	    min_i,max_i,
	    //j-2,j+2,
	    //i-2,i+2,
	    *in_density_cast);
	  
	  find_inverse_and_bck_densels(*kappa0_ptr_bck,all_segments_for_kappa0,
	    all_attenuation_segments,
	    vox_image_ptr_kappa1->get_min_z(),vox_image_ptr_kappa1->get_max_z(),
	    j,j,i,i,
	    *proj_matrix_ptr,false,threshold, true);
	  
	}
	float sq_kapas;
//	float multiply_with_sensitivity;
	if ( fabs((double)(*kappa1_ptr_bck)[k][j][i]) > 0.00000000000001 && 
	  fabs((double)(*kappa0_ptr_bck)[k][j][i]) > 0.00000000000001 )
	{ 
	  sq_kapas =((*kappa0_ptr_bck)[k][j][i]*(*kappa0_ptr_bck)[k][j][i])/((*kappa1_ptr_bck)[k][j][i]*(*kappa1_ptr_bck)[k][j][i]);

	   //cerr << " kappa0 " << (*kappa0_ptr_bck)[k][j][i] << endl;
	   //cerr << " kappa1 " << (*kappa1_ptr_bck)[k][j][i] << endl;



	/*  multiply_with_sensitivity = (*sensitivity_image)[ceil(in_density_cast->get_max_z()-in_density_cast->get_min_z())/2][6][6];
	  if ((*sensitivity_image)[k][j][i] >0.0000001 || (*sensitivity_image)[k][j][i] <-0.0000001)
	  {
	  multiply_with_sensitivity /= (*sensitivity_image)[k][j][i];
	  }
	  else
	  {
	    multiply_with_sensitivity /= 0.000001F;
	  }
	  
	  sq_kapas *= multiply_with_sensitivity;  */
#else
	 float sq_kapas = 1.0F;
#endif
	 (*kappa_coefficients)[k][j][i] = sq_kapas;
	   //sq_kapas = 2;
	  
	
	 
	  //cerr << " now printing sq_kappas value:" << "   " << sq_kapas << endl;
	  int k_index ;
          k_index = round(((float)sq_kapas- k_min)/k_interval);
	   if (k_index < 1) 
	   {k_index = 1;}
	    
	   if ( k_index > num_elements_in_interval)
	   { k_index  = num_elements_in_interval;}
	  
	  
	  if ( filter_lookup[k_index]==NULL )
	  {
	    Array <2,float> new_coeffs;
	    cerr << "\ncomputing new filter for sq_kappas " << sq_kapas << " at index " <<k_index<< std::endl;
	    construct_scaled_filter_coefficients_2D(new_coeffs, filter_coefficients,sq_kapas);    
	    filter_lookup[k_index] = new ArrayFilter2DUsingConvolution<float>(new_coeffs);
	    all_filter_coefficients[k][j][i] = filter_lookup[k_index];
	    //new ArrayFilter3DUsingConvolution<float>(new_coeffs);	
	    
	  }
	  else 
	  {
	    all_filter_coefficients[k][j][i] = filter_lookup[k_index];
		  
	  }
		  
	}
	else
	{	


	  all_filter_coefficients[k][j][i] =
	    new  ArrayFilter2DUsingConvolution<float>();
	  
	}
	
      }   

      
  //write_basic_interfile("kappa_coefficients_2D_SENS",*kappa_coefficients);
  delete kappa_coefficients ;

      
  for (int segment_num = start_segment_num; segment_num <= end_segment_num; ++segment_num)
  {
    delete  all_segments_for_kappa0[segment_num];
  }
    
      
      
}



template <typename elemT>
void 
ModifiedInverseAveragingImageFilterAll<elemT>::precalculate_filter_coefficients_separable(VectorWithOffset < VectorWithOffset < VectorWithOffset <shared_ptr <ModifiedInverseAverigingArrayFilter <3, float> >  > > >& all_filter_coefficients,
									      DiscretisedDensity<3,elemT>* in_density) const
{
 
  VectorWithOffset < shared_ptr <ModifiedInverseAverigingArrayFilter <3, float> > > filter_lookup;
  const int num_elements_in_interval = 500;
  filter_lookup.grow(1,num_elements_in_interval);
  const int k_min =1;
  const float k_interval = 0.01F; //0.01F;
  

  shared_ptr<ProjDataInfo> new_data_info_ptr  = proj_data_ptr->get_proj_data_info_ptr()->clone();
  VoxelsOnCartesianGrid<float>* in_density_cast =
    dynamic_cast< VoxelsOnCartesianGrid<float>* >(in_density); 
  

  VoxelsOnCartesianGrid<float> *  vox_image_ptr_1 =
    new VoxelsOnCartesianGrid<float> (IndexRange3D(in_density_cast->get_min_z(),in_density_cast->get_max_z(),
    in_density_cast->get_min_y(),in_density_cast->get_max_y(),
    in_density_cast->get_min_x(),in_density_cast->get_max_x()),
    in_density_cast->get_origin(),in_density_cast->get_voxel_size());  
  
  int start_segment_num = proj_data_ptr->get_min_segment_num();
  int end_segment_num = proj_data_ptr->get_max_segment_num();
  
  VectorWithOffset<SegmentByView<float> *> all_segments(start_segment_num, end_segment_num);
  VectorWithOffset<SegmentByView<float> *> all_segments_for_kappa0(start_segment_num, end_segment_num);   
  VectorWithOffset<SegmentByView<float> *> all_attenuation_segments(start_segment_num, end_segment_num);
  
  
  // first initialise to false
  bool do_attenuation = false;

  for (int segment_num = start_segment_num; segment_num <= end_segment_num; ++segment_num)
  {
    all_segments[segment_num] = new SegmentByView<float>(proj_data_ptr->get_empty_segment_by_view(segment_num));
    all_segments_for_kappa0[segment_num] = new SegmentByView<float>(proj_data_ptr->get_empty_segment_by_view(segment_num));
    
    if (attenuation_proj_data_filename!="1")
    {
      do_attenuation = true;
      all_attenuation_segments[segment_num] = 
	new SegmentByView<float>(attenuation_proj_data_ptr->get_segment_by_view(segment_num));
    }
    else 
    {
      do_attenuation = false;
      all_attenuation_segments[segment_num] = new SegmentByView<float>(proj_data_ptr->get_empty_segment_by_view(segment_num));		 
      (*all_attenuation_segments[segment_num]).fill(1);
    }
    
  }
  
  vox_image_ptr_1->set_origin(Coordinate3D<float>(0,0,0));   
  
  shared_ptr<DiscretisedDensity<3,float> > image_sptr =  vox_image_ptr_1;
  
  shared_ptr<ProjMatrixByDensel> proj_matrix_ptr = 
    new ProjMatrixByDenselUsingRayTracing;
  
  proj_matrix_ptr->set_up(proj_data_ptr->get_proj_data_info_ptr()->clone(),
    image_sptr);
  cerr << proj_matrix_ptr->parameter_info();
  
  fwd_densels_all(all_segments,proj_matrix_ptr, proj_data_ptr,
    in_density_cast->get_min_z(), in_density_cast->get_max_z(),
    in_density_cast->get_min_y(), in_density_cast->get_max_y(),
    in_density_cast->get_min_x(), in_density_cast->get_max_x(),
    *in_density);
  
  VoxelsOnCartesianGrid<float> *  vox_image_ptr_kappa0 =
    new VoxelsOnCartesianGrid<float>(IndexRange3D(in_density_cast->get_min_z(),in_density_cast->get_max_z(),
    in_density_cast->get_min_y(),in_density_cast->get_max_y(),
    in_density_cast->get_min_x(),in_density_cast->get_max_x()),
    in_density_cast->get_origin(),in_density_cast->get_voxel_size());  
  
  VoxelsOnCartesianGrid<float> *  vox_image_ptr_kappa1 =
    new VoxelsOnCartesianGrid<float>(IndexRange3D(in_density_cast->get_min_z(),in_density_cast->get_max_z(),
    in_density_cast->get_min_y(),in_density_cast->get_max_y(),
    in_density_cast->get_min_x(),in_density_cast->get_max_x()),
    in_density_cast->get_origin(),in_density_cast->get_voxel_size());  

   VoxelsOnCartesianGrid<float> *  kappa_coefficients =
    new VoxelsOnCartesianGrid<float>(IndexRange3D(in_density_cast->get_min_z(),in_density_cast->get_max_z(),
    in_density_cast->get_min_y(),in_density_cast->get_max_y(),
    in_density_cast->get_min_x(),in_density_cast->get_max_x()),
    in_density_cast->get_origin(),in_density_cast->get_voxel_size());
  
  
  shared_ptr<DiscretisedDensity<3,float> > kappa0_ptr_bck =  vox_image_ptr_kappa0;       
  shared_ptr<DiscretisedDensity<3,float> > kappa1_ptr_bck =  vox_image_ptr_kappa1;   
  
  // WARNING - find a way of finding max in the sinogram
  // TODO - include other segments as well
  float max_in_viewgram =0.F;
  
  for (int segment_num = start_segment_num; segment_num<= end_segment_num; segment_num++) 
  {
    SegmentByView<float> segment_by_view = 
      proj_data_ptr->get_segment_by_view(segment_num);
    const float current_max_in_viewgram = segment_by_view.find_max();
    if ( current_max_in_viewgram >= max_in_viewgram)
      max_in_viewgram = current_max_in_viewgram ;
    else
      continue;
  }
  const float threshold = 0.0001F*max_in_viewgram;  
  
  cerr << " THRESHOLD IS" << threshold; 
  cerr << endl;
  
  find_inverse_and_bck_densels(*kappa1_ptr_bck,all_segments,
    all_attenuation_segments,
    vox_image_ptr_kappa1->get_min_z(),vox_image_ptr_kappa1->get_max_z(),
    vox_image_ptr_kappa1->get_min_y(),vox_image_ptr_kappa1->get_max_y(),
    vox_image_ptr_kappa1->get_min_x(),vox_image_ptr_kappa1->get_max_x(),
    *proj_matrix_ptr, do_attenuation,threshold, false); //true);
  
  for (int segment_num = start_segment_num; segment_num <= end_segment_num; ++segment_num)
  { 
    delete all_segments[segment_num];
    delete all_attenuation_segments[segment_num];
  }   
  
  cerr << "min and max in image - kappa1 " <<kappa1_ptr_bck->find_min()
    << ", " << kappa1_ptr_bck->find_max() << endl;   
  
  for (int k=in_density_cast->get_min_z();k<=in_density_cast->get_max_z();k++)   
    for (int j =in_density_cast->get_min_y();j<=in_density_cast->get_max_y();j++)
      for (int i =in_density_cast->get_min_x();i<=in_density_cast->get_max_x();i++)	
      {
	
	// WARNING - only works for segment zero at the moment
	// do the calculation of kappa0 here
	kappa0_ptr_bck->fill(0); 
	for (int segment_num = start_segment_num; 
	segment_num <= end_segment_num; ++segment_num)
	{  
	  (*all_segments_for_kappa0[segment_num]).fill(0);	    
	}
	if (true) //attenuation_proj_data_filename !="1")
	{
#if 1
	   shared_ptr< VoxelsOnCartesianGrid<float> > in_density_cast_tmp =
	   new VoxelsOnCartesianGrid<float>
	    (IndexRange3D(k,k,
	    -mask_size+6,mask_size+6,
	    -mask_size+6,mask_size+6),in_density_cast->get_origin(),in_density_cast->get_voxel_size());  
	 
	   

	  	// SM 23/05/2002 mask now 3D
	  const int min_k = in_density_cast->get_min_z(); //max(in_density_cast->get_min_z(),k-mask_size);
	  const int max_k = in_density_cast->get_max_z(); // min(in_density_cast->get_max_z(),k+mask_size);			
	  const int min_j = max(in_density_cast->get_min_y(),j-mask_size);
	  const int max_j = min(in_density_cast->get_max_y(),j+mask_size);
	  const int min_i = max(in_density_cast->get_min_x(),i-mask_size);
	  const int max_i = min(in_density_cast->get_max_x(),i+mask_size);
	  
	  // the mask size is in 2D only
	  // SM mask now 3D 
	 //for (int k_in =min_k;k_in<=max_k;k_in++)
	  for (int j_in =min_j;j_in<=max_j;j_in++)
	    for (int i_in =min_i;i_in<=max_i;i_in++)	
	    {
	      (*in_density_cast_tmp)[k][j_in-j +6][i_in-i+6] = (*in_density_cast)[k][j_in][i_in];
	      //(*in_density_cast_tmp)[k][j_in-j +6][i_in-i+6] = (*in_density_cast)[k][j_in][i_in];
	    }
	    
	    fwd_densels_all(all_segments_for_kappa0,proj_matrix_ptr, proj_data_ptr,
	      in_density_cast_tmp->get_min_z(), in_density_cast_tmp->get_max_z(),
	      in_density_cast_tmp->get_min_y(),in_density_cast_tmp->get_max_y(),
	      in_density_cast_tmp->get_min_x(),in_density_cast_tmp->get_max_x(),
	      *in_density_cast_tmp);
	    
	    find_inverse_and_bck_densels(*kappa0_ptr_bck,all_segments_for_kappa0,
	      all_attenuation_segments,
	     k,k,
	      //(ceil(vox_image_ptr_kappa1->get_max_z()-vox_image_ptr_kappa1->get_min_z()))/2,ceil((vox_image_ptr_kappa1->get_max_z()-vox_image_ptr_kappa1->get_min_z())/2),
	      //0,0,0,0,
	      6,6,6,6,
	      *proj_matrix_ptr,false,threshold, false) ;//true);	  
	    //(*kappa0_ptr_bck)[k][j][i] = (*kappa0_ptr_bck)[(ceil(vox_image_ptr_kappa1->get_max_z()-vox_image_ptr_kappa1->get_min_z()))/2][6][6];
	    (*kappa0_ptr_bck)[k][j][i] = (*kappa0_ptr_bck)[k][6][6];
	    
	}
	else
	{
	  const int min_j = max(in_density_cast->get_min_y(),j-mask_size);
	  const int max_j = min(in_density_cast->get_max_y(),j+mask_size);
	  const int min_i = max(in_density_cast->get_min_x(),i-mask_size);
	  const int max_i = min(in_density_cast->get_max_x(),i+mask_size);
	  
	  fwd_densels_all(all_segments_for_kappa0,proj_matrix_ptr, proj_data_ptr,
	    in_density_cast->get_min_z(), in_density_cast->get_max_z(),
	    min_j,max_j,
	    min_i,max_i,
	    *in_density_cast);
	  
	  find_inverse_and_bck_densels(*kappa0_ptr_bck,all_segments_for_kappa0,
	    all_attenuation_segments,
	    vox_image_ptr_kappa1->get_min_z(),vox_image_ptr_kappa1->get_max_z(),
	    j,j,i,i,
	    *proj_matrix_ptr,false,threshold, true);
	  
	}
	float sq_kapas;
	if ( fabs((double)(*kappa1_ptr_bck)[k][j][i]) > 0.00000000000001 && 
	  fabs((double)(*kappa0_ptr_bck)[k][j][i]) > 0.00000000000001 )
	{ 
	 // cerr << "kapa0" << (*kappa0_ptr_bck)[k][j][i] << endl;
	  //cerr << "kapa1" << (*kappa1_ptr_bck)[k][j][i] << endl;

	  sq_kapas =((*kappa0_ptr_bck)[k][j][i]*(*kappa0_ptr_bck)[k][j][i])/((*kappa1_ptr_bck)[k][j][i]*(*kappa1_ptr_bck)[k][j][i]);
  
#else
	 float sq_kapas = 10.0F;
#endif
	 (*kappa_coefficients)[k][j][i] = sq_kapas;

	  int k_index ;
          k_index = round(((float)sq_kapas- k_min)/k_interval);
	   if (k_index < 1) 
	   {k_index = 1;}
	    
	   if ( k_index > num_elements_in_interval)
	   { k_index  = num_elements_in_interval;}
	  
	  
	  if ( filter_lookup[k_index]==NULL )
	  {
	   // Array <1,float> new_coeffs;
	    cerr << "\ncomputing new filter for sq_kappas " << sq_kapas << " at index " <<k_index<< std::endl;
	   // construct_scaled_filter_coefficients_1D(new_coeffs, filter_coefficients,sq_kapas);    
	    filter_lookup[k_index] = 
	      new ModifiedInverseAverigingArrayFilter <3, float>(filter_coefficients,sq_kapas);  
	    all_filter_coefficients[k][j][i] = filter_lookup[k_index];
	    //new ArrayFilter3DUsingConvolution<float>(new_coeffs);	
	    
	  }
	  else 
	  {
	    all_filter_coefficients[k][j][i] = filter_lookup[k_index];
		  
	  }
		  
	}
	else
	{	


	  all_filter_coefficients[k][j][i] =
	    new  ModifiedInverseAverigingArrayFilter <3, float>(); //ArrayFilter2DUsingConvolution<float>();
	  
	}
	
      }   

      
  delete kappa_coefficients ;

      
  for (int segment_num = start_segment_num; segment_num <= end_segment_num; ++segment_num)
  {
    delete  all_segments_for_kappa0[segment_num];
  }
    
      
      
}


// densel stuff - > apply
#if 1

template <typename elemT>
void
ModifiedInverseAveragingImageFilterAll<elemT>:: 
virtual_apply(DiscretisedDensity<3,elemT>& out_density, const DiscretisedDensity<3,elemT>& in_density) const
{
  //the first time virtual_apply is called for this object, counter is set to 0
  static int count=0;
  // every time it's called, counter is incremented
  count++;
  cerr << " checking the counter  " << count << endl; 
  
  const VoxelsOnCartesianGrid<float>& in_density_cast_0 =
    dynamic_cast< const VoxelsOnCartesianGrid<float>& >(in_density); 
  
  // the first set is defined for 2d separable case and the second for 3d case -- 
  // depending weather it is 2d or 3d corresponding coefficints are used. 
  static VectorWithOffset < VectorWithOffset < VectorWithOffset <shared_ptr <ArrayFilter3DUsingConvolution <float> >  > > > all_filter_coefficients;
  static VectorWithOffset < VectorWithOffset < VectorWithOffset <shared_ptr <ArrayFilter2DUsingConvolution <float> >  > > > all_filter_coefficients_nonseparable_2D;  
  static VectorWithOffset < VectorWithOffset < VectorWithOffset <shared_ptr <ModifiedInverseAverigingArrayFilter <3, float> >  > > > all_filter_coefficients_separable;

  if (initial_image_filename!="1")
  {  
    if (count ==1)
    {
      if ( num_dim == 3)
      {
	all_filter_coefficients.grow(in_density_cast_0.get_min_z(),in_density_cast_0.get_max_z());    
	for (int k = in_density_cast_0.get_min_z(); k<=in_density_cast_0.get_max_z();k++)
	{
	  all_filter_coefficients[k].grow(in_density_cast_0.get_min_y(),in_density_cast_0.get_max_y());
	  for (int j = in_density_cast_0.get_min_y(); j<=in_density_cast_0.get_max_y();j++)      
	  {
	    (all_filter_coefficients[k])[j].grow(in_density_cast_0.get_min_x(),in_density_cast_0.get_max_x()); 
	  }
	}
     precalculate_filter_coefficients(all_filter_coefficients,initial_image);
      }
      else if ( num_dim ==2)
      {
	all_filter_coefficients_nonseparable_2D.grow(in_density_cast_0.get_min_z(),in_density_cast_0.get_max_z());    
	for (int k = in_density_cast_0.get_min_z(); k<=in_density_cast_0.get_max_z();k++)
	{
	  all_filter_coefficients_nonseparable_2D[k].grow(in_density_cast_0.get_min_y(),in_density_cast_0.get_max_y());
	  for (int j = in_density_cast_0.get_min_y(); j<=in_density_cast_0.get_max_y();j++)      
	  {
	    (all_filter_coefficients_nonseparable_2D[k])[j].grow(in_density_cast_0.get_min_x(),in_density_cast_0.get_max_x()); 
	  }
	}

        if ( precomputed_coefficients_filename!="1")
	{
	  VoxelsOnCartesianGrid<float>* precomputed_coefficients_image_cast =
	      dynamic_cast< VoxelsOnCartesianGrid<float>* >(precomputed_coefficients_image); 
	  cerr << " In here nonseparable" << endl;
	  for ( int k = precomputed_coefficients_image_cast->get_min_z(); k<=precomputed_coefficients_image_cast->get_max_z();k++)
	    for ( int j = precomputed_coefficients_image_cast->get_min_y(); j<=precomputed_coefficients_image_cast->get_max_y();j++)
	      for ( int i = precomputed_coefficients_image_cast->get_min_x(); i<=precomputed_coefficients_image_cast->get_max_x();i++)
	      {
		Array <2,float> new_coeffs;
		//cerr << (*precomputed_coefficients_image)[k][j][i] << "     " << endl;
		if((*precomputed_coefficients_image)[k][j][i] >0.00001 )
		{
		  construct_scaled_filter_coefficients_2D(new_coeffs,filter_coefficients,1/(*precomputed_coefficients_image)[k][j][i]); 
		  all_filter_coefficients_nonseparable_2D[k][j][i] = 
		    new ArrayFilter2DUsingConvolution<float>(new_coeffs);	
		}
		else
		{
		  all_filter_coefficients_nonseparable_2D[k][j][i] = 
		    new ArrayFilter2DUsingConvolution<float>();	
		  
		}
		
		
	      }
	}
	else        
	precalculate_filter_coefficients_2D(all_filter_coefficients_nonseparable_2D,initial_image);
	
      }   
      else
      {
	all_filter_coefficients_separable.grow(in_density_cast_0.get_min_z(),in_density_cast_0.get_max_z());    
	for (int k = in_density_cast_0.get_min_z(); k<=in_density_cast_0.get_max_z();k++)
	{
	  all_filter_coefficients_separable[k].grow(in_density_cast_0.get_min_y(),in_density_cast_0.get_max_y());
	  for (int j = in_density_cast_0.get_min_y(); j<=in_density_cast_0.get_max_y();j++)      
	  {
	    (all_filter_coefficients_separable[k])[j].grow(in_density_cast_0.get_min_x(),in_density_cast_0.get_max_x()); 
	  }
	}
	if ( precomputed_coefficients_filename!="1")
	{
	  cerr << " In here " << endl;
	 for ( int k = 0; k<=1;k++)
	   for ( int j = -26; j<=26;j++)
	     for ( int i = -26; i<=26;i++)
	     {
	     //  cerr << k <<"   "<< j <<"   "<< i <<"   "<< endl;
	       cerr << (*precomputed_coefficients_image)[k][j][i] << "     " << endl;
	       if((*precomputed_coefficients_image)[k][j][i] >0.00001 )
	       {
	       all_filter_coefficients_separable[k][j][i]= 
		new ModifiedInverseAverigingArrayFilter <3, float>(filter_coefficients,1/(*precomputed_coefficients_image)[k][j][i]);
	       }
	       else
	       {
		  all_filter_coefficients_separable[k][j][i]= 
		    new ModifiedInverseAverigingArrayFilter <3, float>();

	       }
	  //construct_scaled_filter_coefficients_2D(new_coeffs,filter_coefficients,(*precomputed_coefficients_image)[k][j][i]); 
	  //all_filter_coefficients_separable[k][j][i] = new ArrayFilter2DUsingConvolution<float>(new_coeffs);
	     }

	}
	else

     precalculate_filter_coefficients_separable(all_filter_coefficients_separable,initial_image);
	

      }
    }
   // else 
   // {
   // }
  }
  else // for initial image
    {
      if (count==1)
      {
	all_filter_coefficients_separable.grow(in_density_cast_0.get_min_z(),in_density_cast_0.get_max_z());
	
	for (int k = in_density_cast_0.get_min_z(); k<=in_density_cast_0.get_max_z();k++)
	{
	  all_filter_coefficients_separable[k].grow(in_density_cast_0.get_min_y(),in_density_cast_0.get_max_y());
	  for (int j = in_density_cast_0.get_min_y(); j<=in_density_cast_0.get_max_y();j++)      
	  {
	    (all_filter_coefficients_separable[k])[j].grow(in_density_cast_0.get_min_x(),in_density_cast_0.get_max_x()); 
	    
	    for (int  i = in_density_cast_0.get_min_x(); i<=in_density_cast_0.get_max_x();i++)      
	    {

	      all_filter_coefficients_separable[k][j][i] =     
		new ModifiedInverseAverigingArrayFilter<3,elemT>;
	      
	    }
	  }
	}
	
      }
      
      
      if ( (count % 20) ==0  /*|| count == 1 */) 
      {
	
	shared_ptr<ProjDataInfo> new_data_info_ptr  = proj_data_ptr->get_proj_data_info_ptr()->clone();
	
	int limit_segments= 0;
	new_data_info_ptr->reduce_segment_range(-limit_segments, limit_segments);
	
	
	VoxelsOnCartesianGrid<float> *  vox_image_ptr_1 =
	  new VoxelsOnCartesianGrid<float> (IndexRange3D(in_density_cast_0.get_min_z(),in_density_cast_0.get_max_z(),
	  in_density_cast_0.get_min_y(),in_density_cast_0.get_max_y(),
	  in_density_cast_0.get_min_x(),in_density_cast_0.get_max_x()),
	  in_density.get_origin(),in_density_cast_0.get_voxel_size());  
	
	int start_segment_num = proj_data_ptr->get_min_segment_num();
	int end_segment_num = proj_data_ptr->get_max_segment_num();
	
	VectorWithOffset<SegmentByView<float> *> all_segments(start_segment_num, end_segment_num);
	VectorWithOffset<SegmentByView<float> *> all_attenuation_segments(start_segment_num, end_segment_num);
	
	// first initialise to false
	bool do_attenuation = false;
	
	for (int segment_num = start_segment_num; segment_num <= end_segment_num; ++segment_num)
	{
	  all_segments[segment_num] = new SegmentByView<float>(proj_data_ptr->get_empty_segment_by_view(segment_num));
	  
	  if (attenuation_proj_data_filename !="1")
	  {
	    do_attenuation = true;
	    all_attenuation_segments[segment_num] = 
	      new SegmentByView<float>(attenuation_proj_data_ptr->get_segment_by_view(segment_num));
	  }
	  else 
	  {
	    do_attenuation = false;
	    all_attenuation_segments[segment_num] = new SegmentByView<float>(proj_data_ptr->get_empty_segment_by_view(segment_num));		 
	    (*all_attenuation_segments[segment_num]).fill(1);
	  }
	  
	}
	
	VectorWithOffset<SegmentByView<float> *> all_segments_for_kappa0(start_segment_num, end_segment_num);
	
	
	for (int segment_num = start_segment_num; segment_num <= end_segment_num; ++segment_num)
	  all_segments_for_kappa0[segment_num] = new SegmentByView<float>(proj_data_ptr->get_empty_segment_by_view(segment_num));
	
	
	vox_image_ptr_1->set_origin(Coordinate3D<float>(0,0,0));   
	
	shared_ptr<DiscretisedDensity<3,float> > image_sptr =  vox_image_ptr_1;
	
	shared_ptr<ProjMatrixByDensel> proj_matrix_ptr = 
	  new ProjMatrixByDenselUsingRayTracing;
	
	proj_matrix_ptr->set_up(proj_data_ptr->get_proj_data_info_ptr()->clone(),
	  image_sptr);
	cerr << proj_matrix_ptr->parameter_info();
	
	fwd_densels_all(all_segments,proj_matrix_ptr, proj_data_ptr,
	  in_density_cast_0.get_min_z(), in_density_cast_0.get_max_z(),
	  in_density_cast_0.get_min_y(), in_density_cast_0.get_max_y(),
	  in_density_cast_0.get_min_x(), in_density_cast_0.get_max_x(),
	  in_density_cast_0);
	
	VoxelsOnCartesianGrid<float> *  vox_image_ptr_kappa0 =
	  new VoxelsOnCartesianGrid<float>(IndexRange3D(in_density_cast_0.get_min_z(),in_density_cast_0.get_max_z(),
	  in_density_cast_0.get_min_y(),in_density_cast_0.get_max_y(),
	  in_density_cast_0.get_min_x(),in_density_cast_0.get_max_x()),
	  in_density.get_origin(),in_density_cast_0.get_voxel_size());  
	
	
	shared_ptr<DiscretisedDensity<3,float> > kappa0_ptr_bck =  vox_image_ptr_kappa0;   
	
	VoxelsOnCartesianGrid<float> *  vox_image_ptr_kappa1 =
	  new VoxelsOnCartesianGrid<float>(IndexRange3D(in_density_cast_0.get_min_z(),in_density_cast_0.get_max_z(),
	  in_density_cast_0.get_min_y(),in_density_cast_0.get_max_y(),
	  in_density_cast_0.get_min_x(),in_density_cast_0.get_max_x()),
	  in_density.get_origin(),in_density_cast_0.get_voxel_size());  
	
	shared_ptr<DiscretisedDensity<3,float> > kappa1_ptr_bck =  vox_image_ptr_kappa1;   
	
	// WARNING - find a way of finding max in the sinogram
	// TODO - include other segments as well
	float max_in_viewgram =0.F;
	
	for (int segment_num = 0; segment_num<= 0;
	segment_num++) 
	{
	  SegmentByView<float> segment_by_view = 
	    proj_data_ptr->get_segment_by_view(segment_num);
	  const float current_max_in_viewgram = segment_by_view.find_max();
	  if ( current_max_in_viewgram >= max_in_viewgram)
	    max_in_viewgram = current_max_in_viewgram ;
	  else
	    continue;
	}
	const float threshold = 0.0001F*max_in_viewgram;  
	
	cerr << " THRESHOLD IS" << threshold; 
	cerr << endl;
	
	find_inverse_and_bck_densels(*kappa1_ptr_bck,all_segments,
	  all_attenuation_segments,
	  vox_image_ptr_kappa1->get_min_z(),vox_image_ptr_kappa1->get_max_z(),
	  vox_image_ptr_kappa1->get_min_y(),vox_image_ptr_kappa1->get_max_y(),
	  vox_image_ptr_kappa1->get_min_x(),vox_image_ptr_kappa1->get_max_x(),
	  *proj_matrix_ptr, do_attenuation,threshold,true);
	
	for (int segment_num = start_segment_num; segment_num <= end_segment_num; ++segment_num)
	{ 
	  delete all_segments[segment_num];
	}   
	
	cerr << "min and max in image - kappa1 " <<kappa1_ptr_bck->find_min()
	  << ", " << kappa1_ptr_bck->find_max() << endl;   
	
	char* file1 = "kappa1";
	//cerr <<"  - Saving " << file1 << endl;
	write_basic_interfile(file1, *kappa1_ptr_bck);
	
	
	const string filename ="kapa0_div_kapa1_pf";
	shared_ptr<iostream> output = 
	  new fstream (filename.c_str(), ios::trunc|ios::in|ios::out|ios::binary);
	
	const string filename1 ="values_of_kapa0_and_kapa1_pf";
	shared_ptr<iostream> output1 =     
	  new fstream (filename1.c_str(), ios::trunc|ios::in|ios::out|ios::binary);
	
	
	if (!*output1)
	  error("Error opening output file %s\n",filename1.c_str());
	
	if (!*output)
	  error("Error opening output file %s\n",filename.c_str());
	
	
	*output << "kapa0_div_kapa1" << endl;
	*output << endl;
	*output << endl;
	*output << "Plane number " << endl;   
	
	int size = filter_coefficients.get_length();
	
	//todo - remove
	const string testing_kappas_att="kappa_att";
	shared_ptr<iostream> output_att =     
	  new fstream (testing_kappas_att.c_str(),ios::trunc|ios::out|ios::binary);
	
	const string testing_kappas_noatt="kappa_noatt";
	shared_ptr<iostream> output_noatt =     
	  new fstream (testing_kappas_noatt.c_str(),ios::trunc|ios::out|ios::binary);
	
	if (!*output_att)
	  error("Error opening output file %s\n",testing_kappas_att.c_str());
	
	if (!*output_noatt)
	  error("Error opening output file %s\n",testing_kappas_noatt.c_str());
	
	
	for (int k=in_density_cast_0.get_min_z();k<=in_density_cast_0.get_max_z();k++)   
	  for (int j =in_density_cast_0.get_min_y();j<=in_density_cast_0.get_max_y();j++)
	    for (int i =in_density_cast_0.get_min_x();i<=in_density_cast_0.get_max_x();i++)	
	    {
	      
	      // WARNING - only works for segment zero at the moment
	      // do the calculation of kappa0 here
	      kappa0_ptr_bck->fill(0); 
	      (*all_segments_for_kappa0[all_segments.get_min_index()]).fill(0);	    
	      if (true) //attenuation_proj_data_filename !="1")
	      {
		
		shared_ptr< VoxelsOnCartesianGrid<float> > in_density_cast_tmp =
		  new VoxelsOnCartesianGrid<float>
		  (IndexRange3D(in_density_cast_0.get_min_z(),in_density_cast_0.get_max_z(),
		  -mask_size+6,mask_size+6,
		  -mask_size+6,mask_size+6),in_density.get_origin(),in_density_cast_0.get_voxel_size());  
		
		const int min_j = max(in_density_cast_0.get_min_y(),j-mask_size);
		const int max_j = min(in_density_cast_0.get_max_y(),j+mask_size);
		const int min_i = max(in_density_cast_0.get_min_x(),i-mask_size);
		const int max_i = min(in_density_cast_0.get_max_x(),i+mask_size);
		
		// the mask size is in 2D only 
	
		for (int j_in =min_j;j_in<=max_j;j_in++)
		  for (int i_in =min_i;i_in<=max_i;i_in++)	
		    
		    (*in_density_cast_tmp)[k][j_in-j +6][i_in-i+6] = in_density_cast_0[k][j_in][i_in];
		  
		  fwd_densels_all(all_segments_for_kappa0,proj_matrix_ptr, proj_data_ptr,
		    in_density_cast_0.get_min_z(), in_density_cast_0.get_max_z(),
		    in_density_cast_tmp->get_min_y(),in_density_cast_tmp->get_max_y(),
		    in_density_cast_tmp->get_min_x(),in_density_cast_tmp->get_max_x(),
		    *in_density_cast_tmp);
		  
		  find_inverse_and_bck_densels(*kappa0_ptr_bck,all_segments_for_kappa0,
		    all_attenuation_segments,
		    vox_image_ptr_kappa1->get_min_z(),vox_image_ptr_kappa1->get_max_z(),
		    //0,0,0,0,
		    6,6,6,6,
		    *proj_matrix_ptr,false,threshold, true);	  
		  (*kappa0_ptr_bck)[k][j][i] = (*kappa0_ptr_bck)[k][6][6];
		  
		  
		  *output_att << k <<"  " << j << "  "<< i << "  "<< (*kappa0_ptr_bck)[k][j][i] << endl;
		  
		  
	      }
	      else
	      {
		const int min_j = max(in_density_cast_0.get_min_y(),j-mask_size);
		const int max_j = min(in_density_cast_0.get_max_y(),j+mask_size);
		const int min_i = max(in_density_cast_0.get_min_x(),i-mask_size);
		const int max_i = min(in_density_cast_0.get_max_x(),i+mask_size);
		
		fwd_densels_all(all_segments_for_kappa0,proj_matrix_ptr, proj_data_ptr,
		  in_density_cast_0.get_min_z(), in_density_cast_0.get_max_z(),
		  min_j,max_j,
		  min_i,max_i,
		  //j-2,j+2,
		  //i-2,i+2,
		  in_density_cast_0);
		
		find_inverse_and_bck_densels(*kappa0_ptr_bck,all_segments_for_kappa0,
		  all_attenuation_segments,
		  vox_image_ptr_kappa1->get_min_z(),vox_image_ptr_kappa1->get_max_z(),
		  j,j,i,i,
		  *proj_matrix_ptr,false,threshold, true);
		
		
		*output_noatt << k <<"  " << j << "  "<< i << "  "<< (*kappa0_ptr_bck)[k][j][i] << endl;
		
	      }
	      //	cerr << "min and max in image - kappa0 " <<kappa0_ptr_bck->find_min()
	      //	<< ", " << kappa0_ptr_bck->find_max() << endl; 
	      
	      char* file0 = "kappa0";
	      write_basic_interfile(file0, *kappa0_ptr_bck);
	      
	      float sq_kapas;
	      
	      if ( fabs((double)(*kappa1_ptr_bck)[k][j][i]) > 0.00000000000001 && 
		fabs((double)(*kappa0_ptr_bck)[k][j][i]) > 0.00000000000001 )
	      { 
		sq_kapas =((*kappa0_ptr_bck)[k][j][i]*(*kappa0_ptr_bck)[k][j][i])/((*kappa1_ptr_bck)[k][j][i]*(*kappa1_ptr_bck)[k][j][i]);
		
		*output1 << " Values of kapa0 and kapa1" << endl;
		*output1<< "for k   "<< k;
		*output1 <<":";
		*output1 << j;
		*output1 <<",";
		*output1 <<i;
		*output1 <<"    ";
		//*output1 <<(*image_sptr_0)[k][j][i];
		*output1 <<(*kappa0_ptr_bck)[k][j][i];
		*output1 << "     ";
		*output1 <<(*kappa1_ptr_bck)[k][j][i];
		*output1 << endl;
		*output<< "for k   "<< k;
		*output <<":";
		*output << j;
		*output <<",";
		*output <<i;
		*output <<"    ";
		*output << sq_kapas;
		*output <<endl;
		
		//sq_kapas = 10;
		inverse_filter = 
		  ModifiedInverseAverigingArrayFilter<3,elemT>(filter_coefficients,sq_kapas);	  
		// construct_scaled_filter_coefficients(new_coeffs, filter_coefficients,sq_kapas);
               // all_filter_coefficients[k][j][i] = 
	//	  new ArrayFilter3DUsingConvolution<elemT>(new_coeffs);	
		all_filter_coefficients_separable[k][j][i] = 
		  new ModifiedInverseAverigingArrayFilter<3,elemT>(inverse_filter);		
		
	      }
	      else
	      {	
		sq_kapas = 0;
		inverse_filter = 
		  ModifiedInverseAverigingArrayFilter<3,elemT>();	  
		all_filter_coefficients_separable[k][j][i] =
		  new ModifiedInverseAverigingArrayFilter<3,elemT>(inverse_filter);
		
	      }
	      
	}      
	
	for (int segment_num = start_segment_num; segment_num <= end_segment_num; ++segment_num)
	{ 
	  delete all_segments_for_kappa0[segment_num];
	  delete all_attenuation_segments[segment_num];
	}   
     }
     }
     if ( initial_image_filename =="1" || num_dim ==1)
     {
     for (int k=in_density_cast_0.get_min_z();k<=in_density_cast_0.get_max_z();k++)   
       for (int j =in_density_cast_0.get_min_y();j<=in_density_cast_0.get_max_y();j++)
	 for (int i =in_density_cast_0.get_min_x();i<=in_density_cast_0.get_max_x();i++)	
	 {
	   Array<3,elemT> tmp_out(IndexRange3D(k,k,j,j,i,i));

	   (*all_filter_coefficients_separable[k][j][i])(tmp_out,in_density);
	   out_density[k][j][i] = tmp_out[k][j][i];	
	   	   
	 }
     }
     else
     {
     
       for (int k=in_density_cast_0.get_min_z();k<=in_density_cast_0.get_max_z();k++)   
	 for (int j =in_density_cast_0.get_min_y();j<=in_density_cast_0.get_max_y();j++)
	   for (int i =in_density_cast_0.get_min_x();i<=in_density_cast_0.get_max_x();i++)	
	   {
	     Array<3,elemT> tmp_out(IndexRange3D(k,k,j,j,i,i));
	     if ( num_dim == 3)
	     {
	     (*all_filter_coefficients[k][j][i])(tmp_out,in_density);
	      out_density[k][j][i] = tmp_out[k][j][i];	
	     
	     }
	     else
	     {
	       Array<2,elemT> single_pixel(IndexRange2D(j,j,i,i));
	       if ( k==in_density_cast_0.get_min_z() && j==in_density_cast_0.get_min_y() 
		 && i==in_density_cast_0.get_min_x() && count == 300)
	       { 
		 cerr <<  " IN the LOOP "  << k << "   " << j << "  " <<i << "   " << endl;
		 for (int k=in_density_cast_0.get_min_z();k<=in_density_cast_0.get_max_z();k++)   
		   for (int j =in_density_cast_0.get_min_y();j<=in_density_cast_0.get_max_y();j++)
		     for (int i =in_density_cast_0.get_min_x();i<=in_density_cast_0.get_max_x();i++)	
		     { 
		       Array <2,float> new_coeffs;
		       if(in_density_cast_0[k][j][i] >0.00001 )
		       {
			 VoxelsOnCartesianGrid<float> *  newly_computed_coeff =
			 new VoxelsOnCartesianGrid<float>(IndexRange3D(in_density_cast_0.get_min_z(),in_density_cast_0.get_max_z(),
			 in_density_cast_0.get_min_y(),in_density_cast_0.get_max_y(),
			 in_density_cast_0.get_min_x(),in_density_cast_0.get_max_x()),
			 in_density.get_origin(),in_density_cast_0.get_voxel_size()); 

			 VoxelsOnCartesianGrid<float> *  normalised_bck_image_cast =
			   dynamic_cast< VoxelsOnCartesianGrid<float> * > (normalised_bck_image);	 			 
			 VoxelsOnCartesianGrid<float> *  sensitivity_image_cast =
			   dynamic_cast< VoxelsOnCartesianGrid<float> * > (sensitivity_image);	 
			
			 
			 precompute_filter_coefficients_for_second_apporach(*newly_computed_coeff,
							in_density_cast_0,
							*sensitivity_image_cast,
							*normalised_bck_image_cast);
			 construct_scaled_filter_coefficients_2D(new_coeffs,filter_coefficients,1/(*newly_computed_coeff)[k][j][i]); 
			 all_filter_coefficients_nonseparable_2D[k][j][i] = 
			   new ArrayFilter2DUsingConvolution<float>(new_coeffs);	
			 delete newly_computed_coeff;
		       }       
		       
		     }
	       }
	       (*all_filter_coefficients_nonseparable_2D[k][j][i])(single_pixel,in_density[k]);
	        out_density[k][j][i] = single_pixel[j][i];
	     }
	    
	     
	   }
	   
     }
     }


#endif



template <typename elemT>
void
ModifiedInverseAveragingImageFilterAll<elemT>:: 
virtual_apply(DiscretisedDensity<3,elemT>& density) const
{
  DiscretisedDensity<3,elemT>* tmp_density =
      density.clone();
  virtual_apply(density, *tmp_density);
  delete tmp_density;
}


template <typename elemT>
void
ModifiedInverseAveragingImageFilterAll<elemT>::set_defaults()
{
  filter_coefficients.fill(0);
  proj_data_filename ="1";
  proj_data_ptr = NULL;
  attenuation_proj_data_filename ="1";
  initial_image_filename ="1";
  sensitivity_image_filename ='1';
  sensitivity_image = NULL;
  precomputed_coefficients_filename ='1';
  normalised_bck_filename ='1';
  normalised_bck_image =NULL;
  precomputed_coefficients_image =NULL;
  attenuation_proj_data_ptr = NULL;
  mask_size = 0;
  num_dim = 1;
 
}

template <typename elemT>
void
ModifiedInverseAveragingImageFilterAll<elemT>:: initialise_keymap()
{
  parser.add_start_key("Modified Inverse Image Filter All Parameters");
  parser.add_key("filter_coefficients", &filter_coefficients_for_parsing);
  parser.add_key("proj_data_filename", &proj_data_filename);
  parser.add_key("attenuation_proj_data_filename", &attenuation_proj_data_filename);
  parser.add_key("initial_image_filename", &initial_image_filename);
  parser.add_key("sensitivity_image_filename", &sensitivity_image_filename);
  parser.add_key("mask_size", &mask_size);
  parser.add_key("num_dim", & num_dim);
  parser.add_key ("precomputed_coefficients_filename", &precomputed_coefficients_filename);
  parser.add_key ("normalised_bck_filename", &normalised_bck_filename);  
  parser.add_stop_key("END Modified Inverse Image Filter All Parameters");
}

template <typename elemT>
bool 
ModifiedInverseAveragingImageFilterAll<elemT>::
post_processing()
{
  const unsigned int size = filter_coefficients_for_parsing.size();
  const int min_index = -(size/2);
  filter_coefficients.grow(min_index, min_index + size - 1);
  for (int i = min_index; i<= filter_coefficients.get_max_index(); ++i)
    filter_coefficients[i] = 
      static_cast<float>(filter_coefficients_for_parsing[i-min_index]);
  return false;
}


const char * const 
ModifiedInverseAveragingImageFilterAll<float>::registered_name =
  "Modified Inverse Image Filter All";


#  ifdef _MSC_VER
// prevent warning message on reinstantiation, 
// note that we get a linking error if we don't have the explicit instantiation below
#  pragma warning(disable:4660)
#  endif

// Register this class in the ImageProcessor registry
// static SeparableCartesianMetzImageFilter<float>::RegisterIt dummy;
// have the above variable in a separate file, which you need t

template ModifiedInverseAveragingImageFilterAll<float>;



END_NAMESPACE_STIR

#endif






