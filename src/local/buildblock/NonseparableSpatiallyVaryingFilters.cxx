//
/*!

  \file
  \ingroup buildblock
  \brief Implementations for class NonseparableSpatiallyVaryingFilters
  
    \author Sanida Mustafovic
    \author Kris Thielemans
    
*/
/*
Copyright (C) 2000- 2003, IRSL
See STIR/LICENSE.txt for details
*/
#include "local/stir/NonseparableSpatiallyVaryingFilters.h"
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
construct_scaled_filter_coefficients_2D(Array<2,float> &new_filter_coefficients_2D_array,   
					Array<2,float>& kernel_2d,const float kapa0_over_kapa1,
					int number_of_coefficients_before_padding);
					


/**********************************************************************************************/





void
construct_scaled_filter_coefficients_2D(Array<2,float> &new_filter_coefficients_2D_array,   
					Array<2,float>& kernel_2d,const float kapa0_over_kapa1,
					int number_of_coefficients_before_padding)
					
{
#if 1
  if (kapa0_over_kapa1!=0)
  {
    
    // in the case where sq_kappas=1 --- scaled_filter == original template filter 
    Array<2,float> filter_coefficients(IndexRange2D(kernel_2d.get_min_index(), kernel_2d.get_max_index(),
      kernel_2d[0].get_min_index(), kernel_2d[0].get_max_index()));
    filter_coefficients = kernel_2d;
    
    // create_kernel_2d (filter_coefficients, kernel_1d);
    
    
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
	
	int filter_length_y = static_cast<int>(floor(kernel_2d.get_length()/2));
	int filter_length_x = static_cast<int>(floor(kernel_2d[0].get_length()/2));
	
	//cerr << "Now doing size " << size << std::endl;
	
	// FIRST PADD 2D FILTER COEFFICIENTS AND MAKE THEM SYMMETRIC 
	// ( DO NOT TAKE IMAGINARY PART INTO ACCOUNT YET)
	/**********************************************************************************/
	
	Array<2,float> filter_coefficients_padded_2D_array(IndexRange2D(1,size,1,size));
	//int number_of_coefficients_before_padding = 15;
	

       int min_kernel_2d_y = kernel_2d.get_min_index();
       int min_kernel_2d_x = kernel_2d[1].get_min_index();
       int tmp_y =  min_kernel_2d_y;
       int tmp_x =  min_kernel_2d_x;	

      for ( int j = 0;j<=number_of_coefficients_before_padding-1;j++)
	for ( int i = 0;i<=number_of_coefficients_before_padding-1;i++)
      {
	filter_coefficients_padded_2D_array[j+1][i+1] = kernel_2d[j+min_kernel_2d_y][i+min_kernel_2d_x];    
	if ( i != (number_of_coefficients_before_padding-1))
	{
	filter_coefficients_padded_2D_array[j+1][size-(i)] = kernel_2d[j+min_kernel_2d_y][i+min_kernel_2d_x+1]; 
	}
	
      }

    int j_n = 1;
    for ( int j = size-(number_of_coefficients_before_padding-1);j<=size-1;j++)
    {  
      for ( int i = 0;i<=number_of_coefficients_before_padding-1;i++)
      {
	filter_coefficients_padded_2D_array[j+1][i+1] = kernel_2d[j_n][i+min_kernel_2d_x];      
	if ( i!=number_of_coefficients_before_padding-1)
	{
	filter_coefficients_padded_2D_array[j+1][size-(i)] = kernel_2d[j_n][i+min_kernel_2d_x+1];  	
	}	
      }
      j_n ++;
    }



    /*   for ( int j = 1;j<=64;j++)
       {
	for ( int i = 1;i<=64;i++)
	{
	  cerr <<  filter_coefficients_padded_2D_array[j][i] << "   " ;
	}
	cerr << endl;
       }
*/
	/*************************************************************************************/
	

	// this is not needed any longer 2D kernel already created
/*	VectorWithOffset < float> kernel_1d_vector;
	kernel_1d_vector.grow(1,size);
	for ( int i = 1; i<= size ;i++)
	  kernel_1d_vector[i] = filter_coefficients_padded_1D_array[i];
	
	Array<2,float> filter_coefficients_padded(IndexRange2D(1,size,1,size));
	
	create_kernel_2d (filter_coefficients_padded, kernel_1d_vector);*/
	Array<2,float> filter_coefficients_padded(IndexRange2D(1,size,1,size));
	filter_coefficients_padded = filter_coefficients_padded_2D_array;
	
 	
	// rescale to DC=1
	float tmp = filter_coefficients_padded.sum();
	filter_coefficients_padded /= filter_coefficients_padded.sum();
      /*  for ( int j = filter_coefficients_padded.get_min_index(); j<=filter_coefficients_padded.get_max_index();j++)
	 for ( int i = 	filter_coefficients_padded.get_min_index(); i<=	filter_coefficients_padded.get_max_index();i++)
	   cerr << filter_coefficients_padded[j][i] << "     ";*/
	
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
	  warning("\nNonseparableSpatiallyVaryingFilters: Cannot do this at the moment -- size is too big'.\n");
	  
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
	  warning("\nNonseparableSpatiallyVaryingFilters: Cannot do this at the moment -- size is too big'.\n");;
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
#if 0
	  for ( int i = fft_filter_num_1D_array.get_min_index(); i<=fft_filter_num_1D_array.get_max_index();i++)
	    cerr << fft_filter_num_1D_array[i] << i  << "   ";
#endif	  
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
 
#if 0
	for ( int j = new_filter_coefficients_2D_array_tmp.get_min_index(); j<=new_filter_coefficients_2D_array_tmp.get_max_index();j++)
	 for ( int i = new_filter_coefficients_2D_array_tmp[1].get_min_index(); i<=new_filter_coefficients_2D_array_tmp[1].get_max_index();i++)
	 {	 
	    cerr << new_filter_coefficients_2D_array_tmp[j][i] << i  << "   ";
	 }
	  
#endif
	
	int kernel_length_x=0;
	int kernel_length_y=0;
	
	// to prevent form aliasing limit the new range for the coefficients to 
	// filter_coefficients_padded.get_length()/2
	
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
	
	if (kernel_length_x == filter_coefficients_padded.get_length()/2)
	{
	  warning("NonseparableSpatiallyVaryingFilters: kernel_length_x reached maximum length %d. "
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
	else if (kernel_length_y == filter_coefficients_padded.get_length()/2)
	{
	  warning("NonseparableSpatiallyVaryingFilters: kernel_length_y reached maximum length %d. "
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
	  

	 // cerr << " IN THIS LOOP " << endl;
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
#if 0
	    for (int  j = 0;j<= kernel_length_y-1;j++)	  
	    for (int  i = 0;i<= kernel_length_x-1;i++)	  
	    {
	      cerr << new_filter_coefficients_2D_array[j][i] << "    " ;
	    }
#endif
	    
	    
	    break; // out of while(true)
	}
    } // this bracket is for the while loop
    }
    else //sq_kappas == 1
    {
	 
	     new_filter_coefficients_2D_array.grow(IndexRange2D(
	    -number_of_coefficients_before_padding-1,number_of_coefficients_before_padding-1,
	    -number_of_coefficients_before_padding-1,number_of_coefficients_before_padding-1));

          for (int  j = 0;j<= number_of_coefficients_before_padding-1;j++)	  
	    for (int  i = 0;i<= number_of_coefficients_before_padding-1;i++)	  		  
	    {
	      new_filter_coefficients_2D_array[j][i]=filter_coefficients[filter_coefficients.get_min_index()+j][filter_coefficients[1].get_min_index()+i];
	      new_filter_coefficients_2D_array[-j][-i]=filter_coefficients[filter_coefficients.get_min_index()+j][filter_coefficients[1].get_min_index()+i];
	      
	      new_filter_coefficients_2D_array[-j][i]=filter_coefficients[filter_coefficients.get_min_index()+j][filter_coefficients[1].get_min_index()+i];
	      new_filter_coefficients_2D_array[j][-i]=filter_coefficients[filter_coefficients.get_min_index()+j][filter_coefficients[1].get_min_index()+i];
	          
	    }
	   

      //new_filter_coefficients_2D_array = filter_coefficients;
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
#endif

}

#if 1


template <typename elemT>
NonseparableSpatiallyVaryingFilters<elemT>::
NonseparableSpatiallyVaryingFilters()
{ 
  set_defaults();
}


template <typename elemT>
NonseparableSpatiallyVaryingFilters<elemT>::
NonseparableSpatiallyVaryingFilters(string proj_data_filename_v,
				       string attenuation_proj_data_filename_v,
				       const Array<2,float>& filter_coefficients_v,
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
NonseparableSpatiallyVaryingFilters<elemT>::
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
NonseparableSpatiallyVaryingFilters<elemT>::precalculate_filter_coefficients_2D (VectorWithOffset < VectorWithOffset < VectorWithOffset <shared_ptr <ArrayFilter2DUsingConvolution <float> >  > > >& all_filter_coefficients,
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
	  
#else
	  float sq_kapas = 1.0F;
#endif
	  (*kappa_coefficients)[k][j][i] = sq_kapas;
	  
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
	    construct_scaled_filter_coefficients_2D(new_coeffs, filter_coefficients,sq_kapas, number_of_coefficients_before_padding);    
	    filter_lookup[k_index] = new ArrayFilter2DUsingConvolution<float>(new_coeffs);
	    all_filter_coefficients[k][j][i] = filter_lookup[k_index];
	    
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
NonseparableSpatiallyVaryingFilters<elemT>:: 
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
  static VectorWithOffset < VectorWithOffset < VectorWithOffset <shared_ptr <ArrayFilter2DUsingConvolution <float> >  > > > all_filter_coefficients_nonseparable_2D;  
  
  if (initial_image_filename!="1")
  {  
    if (count ==1)
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
	      if((*precomputed_coefficients_image)[k][j][i] >0.00001 )
	      {
		construct_scaled_filter_coefficients_2D(new_coeffs,filter_coefficients,1/(*precomputed_coefficients_image)[k][j][i], number_of_coefficients_before_padding); 
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
   
  }
  else // for initial image
  {}
  
  for (int k=in_density_cast_0.get_min_z();k<=in_density_cast_0.get_max_z();k++)   
    for (int j =in_density_cast_0.get_min_y();j<=in_density_cast_0.get_max_y();j++)
      for (int i =in_density_cast_0.get_min_x();i<=in_density_cast_0.get_max_x();i++)	
      {
	Array<3,elemT> tmp_out(IndexRange3D(k,k,j,j,i,i));
	Array<2,elemT> single_pixel(IndexRange2D(j,j,i,i));
	       if ( k==in_density_cast_0.get_min_z() && j==in_density_cast_0.get_min_y() 
		&& i==in_density_cast_0.get_min_x() && count == 300 && precomputed_coefficients_filename!="1" 
		&& normalised_bck_filename !="1")

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
			 construct_scaled_filter_coefficients_2D(new_coeffs,filter_coefficients,1/(*newly_computed_coeff)[k][j][i], number_of_coefficients_before_padding); 
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
#endif
     
     
     
     template <typename elemT>
       void
       NonseparableSpatiallyVaryingFilters<elemT>:: 
       virtual_apply(DiscretisedDensity<3,elemT>& density) const
     {
       DiscretisedDensity<3,elemT>* tmp_density =
	 density.clone();
       virtual_apply(density, *tmp_density);
       delete tmp_density;
     }
     
     
     template <typename elemT>
       void
       NonseparableSpatiallyVaryingFilters<elemT>::set_defaults()
     {
       filter_coefficients.fill(0);
       proj_data_filename ="1";
       proj_data_ptr = NULL;
       attenuation_proj_data_filename ="1";
       initial_image_filename ="1";
       initial_image =NULL;
       sensitivity_image_filename ='1';
       sensitivity_image = NULL;
       precomputed_coefficients_filename ='1';
       normalised_bck_filename ='1';
       normalised_bck_image =NULL;
       precomputed_coefficients_image =NULL;
       attenuation_proj_data_ptr = NULL;
       mask_size = 0;
       num_dim = 1;
       number_of_coefficients_before_padding =0;
       
     }
     
     template <typename elemT>
       void
       NonseparableSpatiallyVaryingFilters<elemT>:: initialise_keymap()
     {
       parser.add_start_key("Nonseparable Spatially Varying Filters");
       parser.add_key("filter_coefficients", &filter_coefficients_for_parsing);
       parser.add_key("proj_data_filename", &proj_data_filename);
       parser.add_key("attenuation_proj_data_filename", &attenuation_proj_data_filename);
       parser.add_key("initial_image_filename", &initial_image_filename);
       parser.add_key("sensitivity_image_filename", &sensitivity_image_filename);
       parser.add_key("mask_size", &mask_size);
       parser.add_key("num_dim", & num_dim);
       parser.add_key ("precomputed_coefficients_filename", &precomputed_coefficients_filename);
       parser.add_key ("normalised_bck_filename", &normalised_bck_filename); 
       parser.add_key("number of coefficients before padding", &number_of_coefficients_before_padding);
       parser.add_stop_key("END Nonseparable Spatially Varying Filters");
     }
     
template <typename elemT>
bool 
NonseparableSpatiallyVaryingFilters<elemT>::
post_processing()
{
  const unsigned int size_x = filter_coefficients_for_parsing[1].get_length();  
  const unsigned int size_y = filter_coefficients_for_parsing.get_length();  

  const int min_index_y = -(size_y/2);
  const int min_index_x = -(size_x/2);
  filter_coefficients.grow(IndexRange2D(min_index_y, min_index_y + size_y - 1,
  				   min_index_x, min_index_x + size_x - 1 ));

  for (int j = min_index_y; j<=filter_coefficients.get_max_index(); ++j)
    for (int i = min_index_x; i<= filter_coefficients[j].get_max_index(); ++i)
    {
    filter_coefficients[j][i] = 
      static_cast<float>(filter_coefficients_for_parsing[j-min_index_y][i-min_index_x]);
    }
return false;
}
     
     
const char * const 
NonseparableSpatiallyVaryingFilters<float>::registered_name =
"Nonseparable Spatially Varying Filters";
   
     
#  ifdef _MSC_VER
     // prevent warning message on reinstantiation, 
     // note that we get a linking error if we don't have the explicit instantiation below
#  pragma warning(disable:4660)
#  endif
     
     // Register this class in the ImageProcessor registry
     // static SeparableCartesianMetzImageFilter<float>::RegisterIt dummy;
     // have the above variable in a separate file, which you need t
     
     template NonseparableSpatiallyVaryingFilters<float>;
     
     
     
     END_NAMESPACE_STIR
       
#endif
