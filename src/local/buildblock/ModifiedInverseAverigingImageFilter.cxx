//
/*!

  \file
  \ingroup buildblock
  \brief Implementations for class ModifiedInverseAverigingImageFilter
  
    \author Sanida Mustafovic
    \author Kris Thielemans
    
*/
/*
    Copyright (C) 2000- 2011, IRSL
    See STIR/LICENSE.txt for details
*/
#include "local/stir/ModifiedInverseAverigingImageFilter.h"
#include "stir/IndexRange3D.h"
#include "stir/shared_ptr.h"
#include "stir/ProjData.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/recon_buildblock/ForwardProjectorByBin.h"
#include "stir/recon_buildblock/BackProjectorByBin.h"
#include "stir/ProjDataFromStream.h"
#include "stir/recon_buildblock/ProjMatrixByBin.h"
#include "local/stir/recon_buildblock/ProjMatrixByDensel.h"
#include "local/stir/recon_buildblock/ProjMatrixByDenselUsingRayTracing.h"
#include "stir/IO/interfile.h"
#include "stir/CartesianCoordinate3D.h"
#include "stir/RelatedViewgrams.h"
#include "local/stir/fft.h"
#include "local/stir/ArrayFilter3DUsingConvolution.h"
#include "local/stir/ArrayFilter2DUsingConvolution.h"


#include "stir/CPUTimer.h"

#include "local/stir/fwd_and_bck_manipulation_for_SAF.h"
#include "stir/recon_buildblock/BackProjectorByBinUsingSquareProjMatrixByBin.h"
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

START_NAMESPACE_STIR


static void
construct_scaled_filter_coefficients(Array<3,float> &new_filter_coefficients_3D_array,   
				     const VectorWithOffset<float>& kernel_1d,
				     const bool z_direction_trivial,				     
				     const float kapa0_over_kapa1);

		  //// IMPLEMENTATION /////
/**********************************************************************************************/



static void
construct_scaled_filter_coefficients(Array<3,float> &new_filter_coefficients_3D_array,   
				     const VectorWithOffset<float>& kernel_1d,
				     const bool z_direction_trivial,
				     const float kapa0_over_kapa1)
				     
{
  
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
  
  
  float sq_kapas = kapa0_over_kapa1; 
  /******************************************************************************************/
  
  if ( sq_kapas > 10000)
  {
    new_filter_coefficients_3D_array.grow(IndexRange3D(0,0,0,0,0,0));
  }
  else if (sq_kapas!=1.F)
  {
    const int kapa0_over_kapa1_interval = 
      min(static_cast<int>(floor(kapa0_over_kapa1/kapa0_over_kapa1_interval_size)),
          length_of_size_array-1);

    while(true)
    {
      const int size = size_for_kapa0_over_kapa1[kapa0_over_kapa1_interval];
      const int size_z = z_direction_trivial ? 1 : size;
      const int size_y = size; 
      const int size_x = size;
      int filter_length = static_cast<int>(floor(kernel_1d.get_length()/2));
      
      cerr << "Now doing size " << size << std::endl;
      
      
      float inverse_sq_kapas;
      if (fabs((double)sq_kapas ) >0.000000000001)
	inverse_sq_kapas = 1/sq_kapas;
      else 
	inverse_sq_kapas = 0;
      
     // static Array<1,float> fft_filter_1D_array_16(1,2*(size_z==1?1:16)*16*16);
     //static Array<1,float> fft_filter_1D_array_32(1,2*(size_z==1?1:32)*32*32);      
      static Array<1,float> fft_filter_1D_array_64(1,2*(size_z==1?1:64)*64*64);
      static Array<1,float> fft_filter_1D_array_128(1,2*(size_z==1?1:128)*128*128);
      static Array<1,float> fft_filter_1D_array_256(1,2*(size_z==1?1:256)*256*256);
      
      Array<1,float>* fft_filter_1D_array_ptr = 0;
      switch (size)
      {
      case 64:
	fft_filter_1D_array_ptr = &fft_filter_1D_array_64;
	break;
      case 128:
	fft_filter_1D_array_ptr = &fft_filter_1D_array_128;
	break;
      case 256:
	fft_filter_1D_array_ptr = &fft_filter_1D_array_256;
	break;
      default:
	error("\nModifiedInverseAveragingImageFilter: Cannot do this at the moment -- size is too big'.\n");
	break;
      }
      Array<1,float>& fft_filter_1D_array = *fft_filter_1D_array_ptr;
           
      Array<1, int> array_lengths(1,3);
      array_lengths[1] = size_z;
      array_lengths[2] = size_y;
      array_lengths[3] = size_x;
      
      if ( fft_filter_1D_array[1] == 0.F)
      {
	// we have to compute it
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
	

	Array<3,float> filter_coefficients_padded(IndexRange3D(1,size_z,1,size_y,1,size_x));


	create_kernel_3d (filter_coefficients_padded, filter_coefficients_padded_1D_array);
	
	// rescale to DC=1
	filter_coefficients_padded /= filter_coefficients_padded.sum();
	
	convert_array_3D_into_1D_array(fft_filter_1D_array,filter_coefficients_padded);
	
	// TODO remove probably
	if (size_z==1)
	{
	  Array<1,int> array_lengths_2d(1,2);
	  array_lengths_2d[1]=array_lengths[2];
	  array_lengths_2d[2]=array_lengths[3];
	  fourn(fft_filter_1D_array, array_lengths_2d, 2,1);
	}
	else
	  fourn(fft_filter_1D_array, array_lengths, 3,1);
	fft_filter_1D_array /= sqrt(static_cast<float>(size_z *size_y*size_x));   
      }	
      /*
      else fft_filter_1D_array[1] != 0 and we have it computed previously
      */
/*
      cerr << " filter coeff after fft" << endl;
       for ( int i = fft_filter_1D_array.get_min_index(); i<= fft_filter_1D_array.get_max_index();i++)
	 cerr << fft_filter_1D_array[i] << "    "; */

    
      Array<3,float> new_filter_coefficients_3D_array_tmp (IndexRange3D(1,size_z,1,size_y,1,size_x));           
      
      // WARNING  -- this only works for the FFT where the convention is that the final result
      // obtained from the FFT is divided with sqrt(N*N*N)   
      // initialise to 0 to prevent from warnings
      //fourn(fft_1_1D_array, array_lengths, 3,1); 
      float  fft_1_1D_array = 1/sqrt(static_cast<float>(size_z*size_y*size_x));

      
      {        
	Array<1,float> fft_filter_num_1D_array = fft_filter_1D_array;	
	fft_filter_num_1D_array *= fft_1_1D_array;	
	// TODO we make a copy for the denominator here, which isn't necessary
	Array<1,float> fft_filter_denom_1D_array = fft_filter_1D_array;	
	fft_filter_denom_1D_array*= (sq_kapas-1);
	// add fft of 1 (but that's a real constant)
	for ( int i = fft_filter_1D_array.get_min_index(); i<=fft_filter_1D_array.get_max_index(); i+=2)
	{
	  fft_filter_denom_1D_array[i] += fft_1_1D_array;
	}
	fft_filter_denom_1D_array /= sq_kapas;

/*	cerr << " filter denominator " << endl;
	for (int i =1;i<=size_z *size_y*size_x;i++)
	{
	  cerr << fft_filter_denom_1D_array[i] << "   ";
	} 
	cerr << endl;*/

	
	divide_complex_arrays(fft_filter_num_1D_array,fft_filter_denom_1D_array);              
	// TODO remove probably
	if (size_z==1)
	{
	  Array<1,int> array_lengths_2d(1,2);
	  array_lengths_2d[1]=array_lengths[2];
	  array_lengths_2d[2]=array_lengths[3];
	  fourn(fft_filter_num_1D_array, array_lengths_2d, 2,-1);
	}
	else
	  fourn(fft_filter_num_1D_array, array_lengths, 3,-1);
	
	
	// make it consistent with mathemematica -- the output of the       
	fft_filter_num_1D_array  /= sqrt(static_cast<double>(size_z *size_y*size_x));
	
#if 0
//	cerr << "for kappa_0_over_kappa_1 " << kapa0_over_kapa1 ;
	cerr << endl;
	for (int i =1;i<=size_z *size_y*size_x;i++)
	{
	  cerr << fft_filter_num_1D_array[i] << "   ";
	}  
#endif
	
	// take the real part only 
	/*********************************************************************************/
	{
	  Array<1,float> real_div_1D_array(1,size_z *size_y*size_x);
	  
	  for (int i=0;i<=(size_z *size_y*size_x)-1;i++)
	    real_div_1D_array[i+1] = fft_filter_num_1D_array[2*i+1];
	  
	  /*********************************************************************************/
	 	  
	  
	  convert_array_1D_into_3D_array(new_filter_coefficients_3D_array_tmp,real_div_1D_array);

#if 0
	  for (int  k = 1;k<= size_z;k++)
	      for (int  j = 1;j<= size_y;j++)	  
		for (int  i = 1;i<= size_x;i++)
		{
		  cerr << new_filter_coefficients_3D_array_tmp[k][j][i] << "   ";

		}
#endif
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
	  for (int i=new_filter_coefficients_3D_array_tmp[kx][jx].get_min_index();
	       i<=new_filter_coefficients_3D_array_tmp[kx][jx].get_max_index()/4;i++)
	  {
	    if (fabs((double)new_filter_coefficients_3D_array_tmp[kx][jx][i])
	      <= new_filter_coefficients_3D_array_tmp[new_filter_coefficients_3D_array_tmp.get_min_index()][new_filter_coefficients_3D_array_tmp[new_filter_coefficients_3D_array_tmp.get_min_index()].get_min_index()][new_filter_coefficients_3D_array_tmp[new_filter_coefficients_3D_array_tmp.get_min_index()][new_filter_coefficients_3D_array_tmp[new_filter_coefficients_3D_array_tmp.get_min_index()].get_min_index()].get_min_index()]*1/1000000) break;
	    else (kernel_length_x)++;
	  }
	  
	  /******************************* Y DIRECTION ************************************/
	  
	  
	  int ky = new_filter_coefficients_3D_array_tmp.get_min_index();
	  int iy = new_filter_coefficients_3D_array_tmp[ky][new_filter_coefficients_3D_array_tmp.get_min_index()].get_min_index();
	  for (int j=new_filter_coefficients_3D_array_tmp[ky].get_min_index();j<=new_filter_coefficients_3D_array_tmp[ky].get_max_index()/4;j++)
	  {
	    if (fabs((double)new_filter_coefficients_3D_array_tmp[ky][j][iy])
	      //= new_filter_coefficients_3D_array_tmp[new_filter_coefficients_3D_array_tmp.get_min_index()][new_filter_coefficients_3D_array_tmp.get_min_index()][new_filter_coefficients_3D_array_tmp.get_min_index()]*1/100000000) break;
	      <= new_filter_coefficients_3D_array_tmp[new_filter_coefficients_3D_array_tmp.get_min_index()][new_filter_coefficients_3D_array_tmp[new_filter_coefficients_3D_array_tmp.get_min_index()].get_min_index()][new_filter_coefficients_3D_array_tmp[new_filter_coefficients_3D_array_tmp.get_min_index()][new_filter_coefficients_3D_array_tmp[new_filter_coefficients_3D_array_tmp.get_min_index()].get_min_index()].get_min_index()]*1/1000000) break;
	    else (kernel_length_y)++;
	  }
	  
	  /********************************************************************************/
	  
	  /******************************* z DIRECTION ************************************/
	  
	  if (!z_direction_trivial)
	  {
	    int jz = new_filter_coefficients_3D_array_tmp.get_min_index();
	    int iz = new_filter_coefficients_3D_array_tmp[jz][new_filter_coefficients_3D_array_tmp.get_min_index()].get_min_index();
	    for (int k=new_filter_coefficients_3D_array_tmp.get_min_index();k<=new_filter_coefficients_3D_array_tmp.get_max_index()/4;k++)
	    {
	      if (fabs((double)new_filter_coefficients_3D_array_tmp[k][jz][iz])
		//<= new_filter_coefficients_3D_array_tmp[new_filter_coefficients_3D_array_tmp.get_min_index()][new_filter_coefficients_3D_array_tmp.get_min_index()][new_filter_coefficients_3D_array_tmp.get_min_index()]*1/100000000) break;
		<= new_filter_coefficients_3D_array_tmp[new_filter_coefficients_3D_array_tmp.get_min_index()][new_filter_coefficients_3D_array_tmp[new_filter_coefficients_3D_array_tmp.get_min_index()].get_min_index()][new_filter_coefficients_3D_array_tmp[new_filter_coefficients_3D_array_tmp.get_min_index()][new_filter_coefficients_3D_array_tmp[new_filter_coefficients_3D_array_tmp.get_min_index()].get_min_index()].get_min_index()]*1/1000000) break;
	      else (kernel_length_z)++;
	    }
	  }
	  else
	    kernel_length_z = 1;
	  
	  /********************************************************************************/
	  
	  if (kernel_length_x == size_x/4)
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
	  else if (kernel_length_y == size_y/4)
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
	  else if (!z_direction_trivial && kernel_length_z == size_z/4)
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
	    new_filter_coefficients_3D_array.grow(IndexRange3D(-(kernel_length_z-1),kernel_length_z-1,
	      -(kernel_length_y-1),kernel_length_y-1,
	      -(kernel_length_x-1),kernel_length_x-1));

	    if ( z_direction_trivial !=1)
	    {
	    for (int  k = 0;k<= kernel_length_z-1;k++)
	      for (int  j = 0;j<= kernel_length_y-1;j++)	  
		for (int  i = 0;i<= kernel_length_x-1;i++)	  
		{
		  new_filter_coefficients_3D_array[k][j][i]=
		    new_filter_coefficients_3D_array[k][j][-i]=
		    new_filter_coefficients_3D_array[k][-j][-i]=
		    new_filter_coefficients_3D_array[k][-j][i]=
		    new_filter_coefficients_3D_array[-k][j][i]=
		    new_filter_coefficients_3D_array[-k][-j][i]=
		    new_filter_coefficients_3D_array[-k][j][-i]=
		    new_filter_coefficients_3D_array[-k][-j][-i]=
		    new_filter_coefficients_3D_array_tmp[k+1][j+1][i+1];		  
		}
	    }
	    else
	    {
	       for (int  j = 0;j<= kernel_length_y-1;j++)	  
		for (int  i = 0;i<= kernel_length_x-1;i++)	  		  
		{
		  new_filter_coefficients_3D_array[0][j][i]=new_filter_coefficients_3D_array_tmp[1][j+1][i+1];
		  new_filter_coefficients_3D_array[0][-j][-i]=new_filter_coefficients_3D_array_tmp[1][j+1][i+1];

		  new_filter_coefficients_3D_array[0][-j][i]=new_filter_coefficients_3D_array_tmp[1][j+1][i+1];
		  new_filter_coefficients_3D_array[0][j][-i]=new_filter_coefficients_3D_array_tmp[1][j+1][i+1];


		  
		}
	    }


  	break; // out of while(true)
	  }
    } // this bracket is for the while loop
    }
    else //sq_kappas == 1
    {
      // in the case where sq_kappas=1 --- scaled_filter == original template filter 
       new_filter_coefficients_3D_array = 
	 Array<3,float>(IndexRange3D(z_direction_trivial? 0 : kernel_1d.get_min_index(), z_direction_trivial? 0 : kernel_1d.get_max_index(),
                                     kernel_1d.get_min_index(), kernel_1d.get_max_index(),
				     kernel_1d.get_min_index(), kernel_1d.get_max_index()));
  
       create_kernel_3d (new_filter_coefficients_3D_array, kernel_1d);  
    }

    /*cerr << " Input kernel" << endl;
     for ( int i = kernel_1d.get_min_index(); i<=kernel_1d.get_max_index(); i++)
       cerr << kernel_1d[i] << "   ";

    for ( int k = 0; k<= new_filter_coefficients_3D_array.get_max_index(); k++)
      for ( int j = 0; j<= new_filter_coefficients_3D_array.get_max_index(); j++)
	for ( int i = 0; i<= new_filter_coefficients_3D_array.get_max_index(); i++)
	{
	  cerr << new_filter_coefficients_3D_array[k][j][i] << "   ";		  
	  
	}*/
	
	
         
    // rescale to DC=1
    new_filter_coefficients_3D_array /= new_filter_coefficients_3D_array.sum();	
     /*for (int  k = new_filter_coefficients_3D_array.get_min_index();k<=new_filter_coefficients_3D_array.get_max_index() ;k++)
	      for (int  j = new_filter_coefficients_3D_array[k].get_min_index();j<= new_filter_coefficients_3D_array[k].get_max_index();j++)	  
		for (int  i = new_filter_coefficients_3D_array[k][j].get_min_index();i<=new_filter_coefficients_3D_array[k][j].get_max_index();i++)
		{
		  cerr << new_filter_coefficients_3D_array[k][j][i] << "   ";

		}*/
    
}


template <typename elemT>
ModifiedInverseAverigingImageFilter<elemT>::
ModifiedInverseAverigingImageFilter()
{ 
  set_defaults();
}


template <typename elemT>
ModifiedInverseAverigingImageFilter<elemT>::
ModifiedInverseAverigingImageFilter(string proj_data_filename_v,
				    string attenuation_proj_data_filename_v,
				    const VectorWithOffset<elemT>& filter_coefficients_v,
				    shared_ptr<ProjData> proj_data_ptr_v,
				    shared_ptr<ProjData> attenuation_proj_data_ptr_v,
				    DiscretisedDensity<3,float>* initial_image_v,
				    DiscretisedDensity<3,float>* sensitivity_image_v,
				    int mask_size_v,bool z_direction_trivial_v)

				    
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
  mask_size= mask_size_v;
  z_direction_trivial = z_direction_trivial_v;
}


template <typename elemT>
Succeeded 
ModifiedInverseAverigingImageFilter<elemT>::
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

    return Succeeded::yes;
  
}


template <typename elemT>
void 
ModifiedInverseAverigingImageFilter<elemT>::precalculate_filter_coefficients (VectorWithOffset < VectorWithOffset < VectorWithOffset <shared_ptr <ArrayFunctionObject <3,float> >  > > >& all_filter_coefficients,
									      DiscretisedDensity<3,elemT>* in_density) const
{

 
  VectorWithOffset < shared_ptr <ArrayFunctionObject <3,float> > > filter_lookup;
  const int num_of_elements_in_interval = 500;
  filter_lookup.grow(1,num_of_elements_in_interval);
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
  
  //fwd_densels_all(all_segments,proj_matrix_ptr, proj_data_ptr,
    //in_density_cast->get_min_z(), in_density_cast->get_max_z(),
    //in_density_cast->get_min_y(), in_density_cast->get_max_y(),
    //in_density_cast->get_min_x(), in_density_cast->get_max_x(),
    //*in_density);

  do_segments_densels_fwd(*in_density_cast,*proj_data_ptr,all_segments,
            in_density_cast->get_min_z(), in_density_cast->get_max_z(),
	    in_density_cast->get_min_y(), in_density_cast->get_max_y(),
	    in_density_cast->get_min_x(), in_density_cast->get_max_x(),
	    *proj_matrix_ptr);
  
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
	  const int new_k_ctr = 
	    z_direction_trivial==1?0:(round((in_density_cast->get_max_z()-in_density_cast->get_min_z())/2.));
	  const int mask_size_z = 
	    z_direction_trivial==1?0:mask_size;

	   shared_ptr< VoxelsOnCartesianGrid<float> > in_density_cast_tmp =
	   new VoxelsOnCartesianGrid<float>
	    (IndexRange3D(-mask_size_z+new_k_ctr,mask_size_z+new_k_ctr,
	    -mask_size+6,mask_size+6,
	    -mask_size+6,mask_size+6),
	    in_density_cast->get_origin(),in_density_cast->get_voxel_size());  
	  CPUTimer timer;
	  timer.start();
	  
	  	// SM 23/05/2002 mask now 3D
	  
	  const int min_k = max(in_density_cast->get_min_z(),k-mask_size_z);
	  const int max_k = min(in_density_cast->get_max_z(),k+mask_size_z);			
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
	      (*in_density_cast_tmp)[k_in-k+new_k_ctr][j_in-j +6][i_in-i+6] = (*in_density_cast)[k_in][j_in][i_in];
	      //(*in_density_cast_tmp)[k][j_in-j +6][i_in-i+6] = (*in_density_cast)[k][j_in][i_in];
	    }
	    
	    fwd_densels_all(all_segments_for_kappa0,proj_matrix_ptr, proj_data_ptr,
	      in_density_cast_tmp->get_min_z(), in_density_cast_tmp->get_max_z(),
	      in_density_cast_tmp->get_min_y(),in_density_cast_tmp->get_max_y(),
	      in_density_cast_tmp->get_min_x(),in_density_cast_tmp->get_max_x(),
	      *in_density_cast_tmp);
	    
	    find_inverse_and_bck_densels(*kappa0_ptr_bck,all_segments_for_kappa0,
	      all_attenuation_segments,
	     new_k_ctr,new_k_ctr,
	      6,6,6,6,
	      *proj_matrix_ptr,false,threshold, false) ;//true);	  
	    (*kappa0_ptr_bck)[k][j][i] = (*kappa0_ptr_bck)[new_k_ctr][6][6];
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
	  //float tmp  = (*kappa0_ptr_bck)[k][j][i];
	  //float tmp1  = (*kappa1_ptr_bck)[k][j][i];
	  //cerr << "kappa_0 " << (*kappa0_ptr_bck)[k][j][i] << endl;
	  //cerr << "kappa_1 " << (*kappa1_ptr_bck)[k][j][i] << endl;


	  sq_kapas =((*kappa0_ptr_bck)[k][j][i]*(*kappa0_ptr_bck)[k][j][i])/((*kappa1_ptr_bck)[k][j][i]*(*kappa1_ptr_bck)[k][j][i]);
	  
#else
	 float sq_kapas = 10.0F;
#endif
	  (*kappa_coefficients)[k][j][i] = sq_kapas;

	 

	 // cerr << " now printing sq_kappas value:" << "   " << sq_kapas << endl;
	  int k_index ;
	    k_index = round(((float)sq_kapas- k_min)/k_interval);
	   if (k_index < 1) 
	   {k_index = 1;}
	    
	   if ( k_index > num_of_elements_in_interval)
	   { k_index  = num_of_elements_in_interval;}
	  
	  
	  if ( filter_lookup[k_index]==NULL )
	  { // first compute the filter and store in filter_loopup
	    //const bool z_direction_trivial= false;//TODO parse
				     
	    Array <3,float> new_coeffs;
	    cerr << "\ncomputing new filter for sq_kappas " << sq_kapas << " at index " <<k_index<< std::endl;
	    construct_scaled_filter_coefficients(new_coeffs, filter_coefficients,z_direction_trivial,sq_kapas);  
	    filter_lookup[k_index] = new ArrayFilter3DUsingConvolution<float>(new_coeffs);	    

	  }
	  all_filter_coefficients[k][j][i] = filter_lookup[k_index];
		    
	}
	else
	{	
	  all_filter_coefficients[k][j][i] =
	    new  ArrayFilter3DUsingConvolution<float>();	  
	}
	
      }   

      
 // write_basic_interfile("kappa_coefficients_2D_SENS",*kappa_coefficients);
  delete kappa_coefficients ;

      
  for (int segment_num = start_segment_num; segment_num <= end_segment_num; ++segment_num)
  {
    delete  all_segments_for_kappa0[segment_num];
  }
    
      
      
}

// densel stuff - > apply

template <typename elemT>
void
ModifiedInverseAverigingImageFilter<elemT>:: 
virtual_apply(DiscretisedDensity<3,elemT>& out_density, const DiscretisedDensity<3,elemT>& in_density) const
{
  //the first time virtual_apply is called for this object, counter is set to 0
  static int count=0;
  // every time it's called, counter is incremented
  count++;
  cerr << " checking the counter  " << count << endl; 
  
  const VoxelsOnCartesianGrid<float>& in_density_cast_0 =
    dynamic_cast< const VoxelsOnCartesianGrid<float>& >(in_density); 
  
  static VectorWithOffset < VectorWithOffset < VectorWithOffset <shared_ptr <ArrayFunctionObject <3,float> >  > > > all_filter_coefficients;
  

  if (initial_image_filename!="1")
  {  
    if (count ==1)
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
   // else 
   // {
   // }
  }
  else // for initial image
    {
      if (count==1)
      {
	all_filter_coefficients.grow(in_density_cast_0.get_min_z(),in_density_cast_0.get_max_z());
	
	for (int k = in_density_cast_0.get_min_z(); k<=in_density_cast_0.get_max_z();k++)
	{
	  all_filter_coefficients[k].grow(in_density_cast_0.get_min_y(),in_density_cast_0.get_max_y());
	  for (int j = in_density_cast_0.get_min_y(); j<=in_density_cast_0.get_max_y();j++)      
	  {
	    (all_filter_coefficients[k])[j].grow(in_density_cast_0.get_min_x(),in_density_cast_0.get_max_x()); 
	    
	    for (int  i = in_density_cast_0.get_min_x(); i<=in_density_cast_0.get_max_x();i++)      
	    {

	      all_filter_coefficients[k][j][i] =     
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
	
	// int size = filter_coefficients.get_length();
	
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
		all_filter_coefficients[k][j][i] = 
		  new ModifiedInverseAverigingArrayFilter<3,elemT>(filter_coefficients,sq_kapas);
		
	      }
	      else
	      {	
		all_filter_coefficients[k][j][i] =
		  new ModifiedInverseAverigingArrayFilter<3,elemT>();
		
	      }
	      
	}      
	
	for (int segment_num = start_segment_num; segment_num <= end_segment_num; ++segment_num)
	{ 
	  delete all_segments_for_kappa0[segment_num];
	  delete all_attenuation_segments[segment_num];
	}   
     }
     }

     for (int k=in_density_cast_0.get_min_z();k<=in_density_cast_0.get_max_z();k++)   
       for (int j =in_density_cast_0.get_min_y();j<=in_density_cast_0.get_max_y();j++)
	 for (int i =in_density_cast_0.get_min_x();i<=in_density_cast_0.get_max_x();i++)	
	 {
	   Array<3,elemT> tmp_out(IndexRange3D(k,k,j,j,i,i));

	   (*all_filter_coefficients[k][j][i])(tmp_out,in_density);
	   out_density[k][j][i] = tmp_out[k][j][i];	
	   	   
	 }
     }

template <typename elemT>
void
ModifiedInverseAverigingImageFilter<elemT>:: 
virtual_apply(DiscretisedDensity<3,elemT>& density) const
{
  DiscretisedDensity<3,elemT>* tmp_density =
      density.clone();
  virtual_apply(density, *tmp_density);
  delete tmp_density;
}


template <typename elemT>
void
ModifiedInverseAverigingImageFilter<elemT>::set_defaults()
{
  filter_coefficients.fill(0);
  proj_data_filename ="1";
  proj_data_ptr = NULL;
  attenuation_proj_data_filename ="1";
  initial_image_filename ="1";
  sensitivity_image_filename ='1';
  sensitivity_image = NULL;
  attenuation_proj_data_ptr = NULL;
  mask_size = 0;
  z_direction_trivial = true;
 
}

template <typename elemT>
void
ModifiedInverseAverigingImageFilter<elemT>:: initialise_keymap()
{
  parser.add_start_key("Modified Inverse Image Filter Parameters");
  parser.add_key("filter_coefficients", &filter_coefficients_for_parsing);
  parser.add_key("proj_data_filename", &proj_data_filename);
  parser.add_key("attenuation_proj_data_filename", &attenuation_proj_data_filename);
  parser.add_key("initial_image_filename", &initial_image_filename);
  parser.add_key("sensitivity_image_filename", &sensitivity_image_filename);
  parser.add_key("mask_size", &mask_size);
  parser.add_key("z_direction_trivial", &z_direction_trivial);
  parser.add_stop_key("END Modified Inverse Image Filter Parameters");
}

template <typename elemT>
bool 
ModifiedInverseAverigingImageFilter<elemT>::
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
ModifiedInverseAverigingImageFilter<float>::registered_name =
  "Modified Inverse Image Filter";


#  ifdef _MSC_VER
// prevent warning message on reinstantiation, 
// note that we get a linking error if we don't have the explicit instantiation below
#  pragma warning(disable:4660)
#  endif

// Register this class in the ImageProcessor registry
// static SeparableCartesianMetzImageFilter<float>::RegisterIt dummy;
// have the above variable in a separate file, which you need t

template ModifiedInverseAverigingImageFilter<float>;



END_NAMESPACE_STIR


/************************************************************************************************/





