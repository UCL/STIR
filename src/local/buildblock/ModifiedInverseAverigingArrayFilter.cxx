/*
    Copyright (C) 2001- 2002, IRSL
    See STIR/LICENSE.txt for details
*/
#include "local/stir/ModifiedInverseAverigingArrayFilter.h"
#include "stir/ArrayFilter1DUsingConvolution.h"
#include "stir/Array.h"
#include "stir/IndexRange3D.h"
#include "local/stir/fft.h"

#include "local/stir/local_helping_functions.h"
#include <iostream>
#include <fstream>

#ifndef STIR_NO_NAMESPACES
using std::ios;
using std::fstream;
using std::iostream;
using std::cerr;
using std::endl;
#endif



START_NAMESPACE_STIR



void 
FFT_routines::find_fft_filter(Array<1,float>& filter_coefficients)
{
  four1(filter_coefficients,filter_coefficients.get_length()/2,1); 
}
  
void
FFT_routines::find_fft_unity(Array<1,float>& unity)
{
  four1(filter_coefficients,filter_coefficients.get_length()/2,1);
}


template <int num_dimensions, typename elemT>
ModifiedInverseAverigingArrayFilter<num_dimensions,elemT>:: 
ModifiedInverseAverigingArrayFilter()
  : filter_coefficients(0), kapa0_over_kapa1(0)
{ 
  /*filter_coefficients.grow(0,2);
  filter_coefficients[0] =0;  
  filter_coefficients[1] =1;  
  filter_coefficients[2] =0;  */
  
  // because there is no filtering at all, we might as well ignore 3rd direction
  for (int i=2;i<=num_dimensions;i++)
  {
    
    all_1d_array_filters[i-1] = 	       
	new ArrayFilter1DUsingConvolution<float>();//filter_coefficients);
  }

}


//// LOOK UP TABLE VERSION
#if 1
template <int num_dimensions, typename elemT>
ModifiedInverseAverigingArrayFilter<num_dimensions,elemT>::
ModifiedInverseAverigingArrayFilter(const VectorWithOffset<elemT>& kernel_1d,
				    const float kapa0_over_kapa1_v)
				    :
filter_coefficients(kernel_1d),
kapa0_over_kapa1(kapa0_over_kapa1_v)
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
  VectorWithOffset<float> new_filter_coefficients;
  /******************************************************************************************/
  
  if ( sq_kapas > 10000)
  {
    new_filter_coefficients.grow(0,0);
  }
  else if (sq_kapas!=1.F)
  {
    const int kapa0_over_kapa1_interval = 
      min(static_cast<int>(floor(kapa0_over_kapa1/kapa0_over_kapa1_interval_size)),
      length_of_size_array-1);
    
    while(true)
    {
      const int size = size_for_kapa0_over_kapa1[kapa0_over_kapa1_interval];
      int filter_length = static_cast<int>(floor(kernel_1d.get_length()/2));
      
      //cerr << "Now doing size " << size << std::endl;
      
      float inverse_sq_kapas;
      if (fabs((double)sq_kapas ) >0.000000000001)
	inverse_sq_kapas = 1/sq_kapas;
      else 
	inverse_sq_kapas = 0;
      
      static Array<1,float> fft_filter_1D_array_64(1,64);
      static Array<1,float> fft_filter_1D_array_128(1,128);
      static Array<1,float> fft_filter_1D_array_256(1,256);
      static Array<1,float> fft_filter_1D_array_512(1,512);
      static Array<1,float> fft_filter_1D_array_1024(1,1024);
      static Array<1,float> fft_filter_1D_array_2048(1,2048);
      static Array<1,float> fft_filter_1D_array_4096(1,4096);
      static Array<1,float> fft_filter_1D_array_8192(1,8192);

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
      case 512:
	fft_filter_1D_array_ptr = &fft_filter_1D_array_512;
	break;
      case 1024:
	fft_filter_1D_array_ptr = &fft_filter_1D_array_1024;
	break; 
      case 2048:
	fft_filter_1D_array_ptr = &fft_filter_1D_array_2048;
	break;
      case 4096:
	fft_filter_1D_array_ptr = &fft_filter_1D_array_4096;
	break;
      case 8192:
	fft_filter_1D_array_ptr = &fft_filter_1D_array_8192;
	break;
      default:
	error("\nModifiedInverseAveragingImageFilter: Cannot do this at the moment -- size is too big'.\n");
	break;
      }
      Array<1,float>& fft_filter_1D_array = *fft_filter_1D_array_ptr;
      
      
      if ( fft_filter_1D_array[1] == 0.F)
      {
	// we have to compute it
	// FIRST PADD 1D FILTER COEFFICIENTS AND MAKE THEM SYMMETRIC 
	// ( TAKE IMAGINARY PART INTO ACCOUNT YET)
	/**********************************************************************************/
	
	Array<1,float> filter_coefficients_padded_1D_array(1,size);	
	filter_coefficients_padded_1D_array[1] = kernel_1d[0];  
	for ( int i = 1;i<=filter_length;i++)
	{
	  filter_coefficients_padded_1D_array[2*i+1] = filter_coefficients[i];    
	  filter_coefficients_padded_1D_array[size-(2*(i-1)+1)] = filter_coefficients[i];
	  
	}

	/*************************************************************************************/
	//Array<1,float> filter_coefficients_padded(1,size); //1,size_y,1,size_x));	
	//filter_coefficients_padded /= filter_coefficients_padded.sum();
	fft_filter_1D_array = filter_coefficients_padded_1D_array;
	
	four1(fft_filter_1D_array,fft_filter_1D_array.get_length()/2,1);
	
	fft_filter_1D_array /= sqrt(static_cast<float>(size/2));   
      }

        // WARNING  -- this only works for the FFT where the convention is that the final result
	// obtained from the FFT is divided with sqrt(N*N*N)   
	// initialise to 0 to prevent from warnings
	//fourn(fft_1_1D_array, array_lengths, 3,1); 
	float  fft_1_1D_array = 1/sqrt(static_cast<float>(size/2));

	//cerr << " THE unity stuff " << endl;
        //cerr << fft_1_1D_array << "   ";

	 Array<1,float> real_div_1D_array(1,size);
	
	{   
	
	  Array<1,float> fft_filter_num_1D_array(1,fft_filter_1D_array.get_length()); //= fft_filter_1D_array;	
	  fft_filter_num_1D_array = fft_filter_1D_array* fft_1_1D_array;	
	  
	 // TODO we make a copy for the denominator here, which isn't necessary
	  Array<1,float> fft_filter_denom_1D_array(1,fft_filter_1D_array.get_length()); //= fft_filter_1D_array;	
	  fft_filter_denom_1D_array =fft_filter_1D_array*(sq_kapas-1);
	  // add fft of 1 (but that's a real constant)
	  for ( int i = fft_filter_1D_array.get_min_index(); i<=fft_filter_1D_array.get_max_index(); i+=2)
	  {
	    fft_filter_denom_1D_array[i] += fft_1_1D_array;
	  }
	  fft_filter_denom_1D_array /= sq_kapas;
	  
          divide_complex_arrays(fft_filter_num_1D_array,fft_filter_denom_1D_array);              	  
	  four1(fft_filter_num_1D_array,fft_filter_num_1D_array.get_length()/2,-1);

	  
	  
	  // make it consistent with mathemematica -- the output of the       
	  fft_filter_num_1D_array  /= sqrt(static_cast<double>(size/2));
	  

	  // take the real part only 
	  /*********************************************************************************/
	  {   
	    
	    for (int i=0;i<=(size)/2-1;i++)
	      real_div_1D_array[i+1] = fft_filter_num_1D_array[2*i+1];
	    
	    /*********************************************************************************/
	 // cerr << " num " << endl;
	   //for ( int i = real_div_1D_array.get_min_index(); i<=real_div_1D_array.get_max_index();i++)
	     //cerr << real_div_1D_array[i] << "   ";
	    
	  }
	}
	
	int kernel_length=0;
	
	// to prevent form aliasing limit the new range for the coefficients to 
	// filter_coefficients_padded.get_length()/4
	
	
	// do the x -direction first -- fix y and z to the min and look for the max index in x
  	for (int i=real_div_1D_array.get_min_index(); i<=real_div_1D_array.get_max_index()/4;i++)
	       {
		 if (fabs((double)real_div_1D_array[i])
		   <= real_div_1D_array[real_div_1D_array.get_min_index()]*1/1000000) break;
		 else (kernel_length)++;
	       }
	       
	       
	       if (kernel_length == size/4)
	       {
		 warning("ModifiedInverseAverigingArrayFilter3D: kernel_length_x reached maximum length %d. "
		   "First filter coefficient %g, last %g, kappa0_over_kappa1 was %g\n"
		   "Increasing length of FFT array to resolve this problem\n",
		   kernel_length, real_div_1D_array[real_div_1D_array.get_min_index()], real_div_1D_array[kernel_length],
		   kapa0_over_kapa1);
		 size_for_kapa0_over_kapa1[kapa0_over_kapa1_interval]*=2;
		 for (int i=kapa0_over_kapa1_interval+1; i<size_for_kapa0_over_kapa1.get_length(); ++i)
		   size_for_kapa0_over_kapa1[i]=
		   max(size_for_kapa0_over_kapa1[i], size_for_kapa0_over_kapa1[kapa0_over_kapa1_interval]);
	       }
	       else
	       {	 
		 new_filter_coefficients.grow(-(kernel_length-1),kernel_length-1);    		 
		 new_filter_coefficients[0] = real_div_1D_array[1];
		 
		 for (int  i = 1;i<= kernel_length-1;i++)	   
		 {
		   new_filter_coefficients[i]=real_div_1D_array[i+1];
		   new_filter_coefficients[-i]=real_div_1D_array[i+1];
		   
		 }

		
	       
	       break; // out of while(true)
	       }
	  //}
    } // this bracket is for the while loop
    }
    else //sq_kappas == 1
    {
      new_filter_coefficients = kernel_1d;
    }
    
    
    
    float sum_new_coefficients =0.F;  
    for (int i=new_filter_coefficients.get_min_index();i<=new_filter_coefficients.get_max_index();i++)
      sum_new_coefficients += new_filter_coefficients[i];  
    
    for (int i=new_filter_coefficients.get_min_index();i<=new_filter_coefficients.get_max_index();i++)
      new_filter_coefficients[i] /=sum_new_coefficients;  
     
    // to do only filtering in 2d -> 
    // z-direction is for 0 index
    
    for (int i=2;i<=num_dimensions;i++)
    {
      all_1d_array_filters[i-1] = 
	new ArrayFilter1DUsingConvolution<float>(new_filter_coefficients);
    }
    
      }
      
#endif 
      


// SM new 17/09/2001
#if 0
template <int num_dimensions, typename elemT>
ModifiedInverseAverigingArrayFilter<num_dimensions,elemT>::
ModifiedInverseAverigingArrayFilter(const VectorWithOffset<elemT>& filter_coefficients_v,
				    const float kapa0_over_kapa1_v)
  :
  filter_coefficients(filter_coefficients_v),
  kapa0_over_kapa1(kapa0_over_kapa1_v)
{
  
  const string coeff="coeff";
    shared_ptr<iostream> coeff_output =     
      new fstream (coeff.c_str(),ios::out|ios::binary);

  // pipe coefficients to the file to check if they were read correctely
  for (int i=filter_coefficients.get_min_index();i<=filter_coefficients.get_max_index();i++)
  *coeff_output << filter_coefficients[i] << "   ";  
  *coeff_output<< endl;

  //cerr <<kapa0_over_kapa1<< endl;
#if 1
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
/*
  int size;
  // for larger kappa0/kappa1
  if (kapa0_over_kapa1_v >100)
    size = 1024;
  else if (kapa0_over_kapa1_v >50)
    size = 512;
    //size =256;
    else
    size = 256;
    //size =128;  
*/
  //TODO CHECK
    
  /* cerr << "Now printing the value of kapas" << endl;
  cerr << kapa0_over_kapa1_v << endl;*/
  float sq_kapas = kapa0_over_kapa1_v; 
  VectorWithOffset<float> new_filter_coefficients;
  
  if ( sq_kapas > 10000)
  {
    new_filter_coefficients.grow(0,2);
    new_filter_coefficients[0]=0;
    new_filter_coefficients[1]=1;
    new_filter_coefficients[2]=0;
  }
  else if (sq_kapas!=1.F)
  {
    
    while(true)
    {
      const int size = size_for_kapa0_over_kapa1[kapa0_over_kapa1_interval];
      
      int filter_length = static_cast<int>(floor(filter_coefficients.get_length()/2));
      //cerr << " FILTER LENGTH IS " << filter_length << endl;
      
      VectorWithOffset<float> filter_coefficients_padded(1,size);
      filter_coefficients_padded.fill(0);
      // SM 10/12/2001 changed such that all filter sizes could be handeled.  
      filter_coefficients_padded[1] = filter_coefficients[0];  
      for ( int i = 1;i<=filter_length;i++)
      {
	filter_coefficients_padded[2*i+1] = filter_coefficients[i];    
	filter_coefficients_padded[size-(2*(i-1)+1)] = filter_coefficients[i];
      }
      

      
      /*for (int i=1;i<=size;i++)
	  cerr << filter_coefficients_padded[i] << "   ";  
      cerr << endl;*/
      // rescale to DC=1
      float sum =0;  
      for (int i=1;i<=size;i++)
	sum += filter_coefficients_padded[i];  
      
      for (int i=1;i<=size;i++)
	filter_coefficients_padded[i] =filter_coefficients_padded[i]/sum;

     /* for (int i=1;i<=size;i++)
	  cerr << filter_coefficients_padded[i] << "   ";  
      cerr << endl;*/
      
      
      Array<1,float> fft_filter(1,filter_coefficients_padded.get_length());
      Array<1,float> fft_1(1,filter_coefficients_padded.get_length());  
      
      Array<1,float> fft_filter_denom(1,filter_coefficients_padded.get_length());
      Array<1,float> fft_filter_denom_1(1,filter_coefficients_padded.get_length());
      
      
      Array<1,float> fft_filter_num(1,filter_coefficients_padded.get_length());
      Array<1,float> div(1,filter_coefficients_padded.get_length());
      
      fft_filter_denom.fill(0);
      fft_filter_num.fill(0);
      div.fill(0);  
      fft_1[1] =1;
      
      
      for (int i =1;i<=size;i++)
	fft_filter[i] = filter_coefficients_padded[i]; 
      
      
      float inverse_sq_kapas;
      if (fabs((double)sq_kapas ) >0.000000000001)
	inverse_sq_kapas = 1/sq_kapas;
      else 
	inverse_sq_kapas = 0;
      
      fft_filter_denom =fft_filter*(sq_kapas-1) + fft_1;
      fft_filter_denom *= inverse_sq_kapas;
      
      
      four1(fft_filter_denom,fft_filter_denom.get_length()/2,1);
      four1(fft_1,fft_1.get_length()/2,1);
      four1(fft_filter,fft_filter.get_length()/2,1);  
      
      // to check the outputs make the fft consistant with mathematica
      // divide 1/sqrt(size/2)
      for (int i=1; i<=size; i++)
      {
	fft_filter[i] =fft_filter[i]/sqrt(static_cast<double> (size/2));
	fft_1[i] =fft_1[i]/sqrt(static_cast<double> (size/2));
	fft_filter_denom[i] = fft_filter_denom[i]/sqrt(static_cast<double>(size/2));
      }
      
      mulitply_complex_arrays(fft_filter_num,fft_filter,fft_1);
      divide_complex_arrays(div,fft_filter_num,fft_filter_denom);   
      four1(div,div.get_length()/2,-1);
      
      for (int i = div.get_min_index();i<=div.get_max_index();i++)
      {
	div[i] = div[i]/sqrt(static_cast<double> (div.get_length()/2));
      }
      
      Array<1,float> real_div(1,filter_coefficients_padded.get_length()/2);
      real_div[1] = div[1];
      for (int i=1;i<=(size/2)-1;i++)
	real_div[i+1] = div[2*i+1];
      
      int kernel_length=0;
      
      // new - to prevent form aliasing limit the new range for the coefficients to 
      // filter_coefficients_padded.get_length()/4
      // for (int i=1;i<=filter_coefficients_padded.get_length()/2;i++)
      for (int i=1;i<=filter_coefficients_padded.get_length()/4;i++)
      { 
	// SM TESTING SMALLER THRESHOLD - 24/03/2002
	//if (fabs((double) real_div[i])<= real_div[real_div.get_min_index()]*1/1000000000000000) break;
	// SM TESTING SMALLER THRESHOLD - 17/04/2002
	if (fabs((double) real_div[i])<= real_div[real_div.get_min_index()]*1/100000000) break;
	//if (fabs((double) real_div[i])<= real_div[real_div.get_min_index()]*1/100000) break;
     // if (fabs((double) real_div[i])<= real_div[real_div.get_min_index()]*1/1000) break;

	//sm 16/11/2001 try the new threshold
	//if (fabs((double) real_div[i])<= real_div[real_div.get_min_index()]*1/100) break;
	//if (fabs((double) real_div[i])<= real_div[real_div.get_min_index()]*1/10) break;
	else (kernel_length)++;
      }  
      if (kernel_length == filter_coefficients_padded.get_length()/4)
      {
	warning("ModifiedInverseAverigingArrayFilter: kernel_length reached maximum length %d. "
	  "First filter coefficient %g, last %g, kappa0_over_kappa1 was %g\n"
	  "Increasing length of FFT array to resolve this problem\n",
	  kernel_length, real_div[real_div.get_min_index()], real_div[kernel_length],
	  kapa0_over_kapa1);
	size_for_kapa0_over_kapa1[kapa0_over_kapa1_interval]*=2;
	for (int i=kapa0_over_kapa1_interval+1; i<size_for_kapa0_over_kapa1.get_length(); ++i)
	  size_for_kapa0_over_kapa1[i]=
	    max(size_for_kapa0_over_kapa1[i], size_for_kapa0_over_kapa1[kapa0_over_kapa1_interval]);
      }
      else
      {	  
	// new
	//if (kernel_length == real_div.get_length())
	new_filter_coefficients.grow(-(kernel_length-1),kernel_length);    
	//else
	//new_filter_coefficients.grow(-kernel_length,kernel_length);    
	
	new_filter_coefficients[0] = real_div[1];
	new_filter_coefficients[kernel_length] = real_div[kernel_length];
	
	for (int  i = 1;i<= kernel_length-1;i++)
	  
	  //min(15,kernel_length);i++)
	{
	  new_filter_coefficients[i]=real_div[i+1];
	  new_filter_coefficients[-i]=real_div[i+1];
	  
	}
	
	break; // out of while(true)
      }
    } // this bracket is for the while loop
    }
    else //sq_kappas == 1
    {
      new_filter_coefficients = filter_coefficients;
    }
    
      
#endif 
      //cerr << " COEFF PRINT NOW" << endl;
      //for (int i=new_filter_coefficients.get_min_index();i<=new_filter_coefficients.get_max_index();i++)
      //cerr << new_filter_coefficients[i] << "   ";   
      
      const string filename ="coeff_SA_2D_pf_new";
      shared_ptr<iostream> output = new fstream (filename.c_str(), ios::ate|ios::out|ios::binary);
      if (!*output)
	error("Error opening output file %s\n",filename.c_str()); 
      
      // now rescaled the calculated coefficients to DC gain 1
      
      float sum_new_coefficients =0.F;  
      for (int i=new_filter_coefficients.get_min_index();i<=new_filter_coefficients.get_max_index();i++)
	sum_new_coefficients += new_filter_coefficients[i];  
      
      for (int i=new_filter_coefficients.get_min_index();i<=new_filter_coefficients.get_max_index();i++)
	new_filter_coefficients[i] /=sum_new_coefficients;  
      
      *output << "coeff" << endl;
      *output << endl;  
      for (int i=new_filter_coefficients.get_min_index();i<=new_filter_coefficients.get_max_index();i++)
	*output << new_filter_coefficients[i] << "   ";
      *output << endl;
      
     /* cerr << " PRINTING NOW" << endl;
       cerr << " COEFF" << endl;
      for (int i=0;i<=new_filter_coefficients.get_max_index();i++)
      cerr << new_filter_coefficients[i] << "   ";*/
      
      // to do only filtering in 2d -> 
      // z-direction is for 0 index
      kernel_index_range =
	IndexRange3D(0,0, 
	new_filter_coefficients.get_min_index(), new_filter_coefficients.get_max_index(),
	new_filter_coefficients.get_min_index(), new_filter_coefficients.get_max_index());
      
      for (int i=1;i<=num_dimensions;i++)
      {
	all_1d_array_filters[i-1] = 
	  new ArrayFilter1DUsingConvolution<float>(new_filter_coefficients);
      }
      
      //all_1d_array_filters[0] = 	 
//	new ArrayFilter1DUsingConvolution<float>();
      
}


#endif


template ModifiedInverseAverigingArrayFilter<3, float>;

END_NAMESPACE_STIR
























//old

#if 0

template <int num_dimensions, typename elemT>
ModifiedInverseAverigingArrayFilter<num_dimensions,elemT>::
ModifiedInverseAverigingArrayFilter(VectorWithOffset <float>& kapa0_over_kapa1)
{   
  // default coefficients
  // TODO 
  VectorWithOffset<float> denominator_filter_coefficients(0,2);
  denominator_filter_coefficients[0] =0.25;
  denominator_filter_coefficients[1] =0.5;
  denominator_filter_coefficients[2] =0.25;
  
  VectorWithOffset<float>  denominator_coeff_mult(3);
  int min_index = kapa0_over_klapa1.get_min_index();
  int max_index = kapa0_over_klapa1.get_max_index();
  
  VectorWithOffset <float> square_kapa0_over_klapa1(min_index,max_index);
  VectorWithOffset <float> square_kapa1_over_klapa0(min_index,max_index);
  
  VectorWithOffset<VectorWithOffset<float> > vector_denominator_filter_coefficients(min_index, max_index);
  //vector_denominator_filter_coefficients.fill(0);
  
  VectorWithOffset<VectorWithOffset<float> > vector_numerator_filter_coefficients(min_index, max_index);
  //vector_nominator_filter_coefficients.fill(0); 
  
  // grow inner index
  for (int i=min_index;i<=max_index;i++)
  {
    vector_denominator_filter_coefficients[i].grow(vector_denominator_filter_coefficients[i].get_min_index(),vector_denominator_filter_coefficients[i].get_max_index());
    vector_numerator_filter_coefficients[i].grow(vector_numerator_filter_coefficients[i].get_min_index(),vector_numerator_filter_coefficients[i].get_max_index());
  }
  
  for ( int i = min_index;i<=max_index;i++)
  {
    square_kapa0_over_klapa1[i] = (kapa0_over_kapa1[i])*(kapa0_over_kapa1[i]);
  }
  
  for ( int i = min_index;i<=max_index;i++)
  {
    square_kapa0_over_klapa1[i] = 1- square_kapa0_over_klapa1[i];
  }
  
  for (int i =min_index;i<=max_index;i++)
    for(int j=0;j<=2;j++)
    {
      vector_denominator_filter_coefficients[i][j] = denominator_filter_coefficients[i]*square_kapa0_over_klapa1[i];
      vector_denominator_filter_coefficients[i][j] *=-1;    
      vector_denominator_filter_coefficients[i][j] +=1;
      
    }
    
    // multiply with square(kapa1/kapa0)
    for (int i =min_index;i<=max_index;i++)
      for(int j=0;j<=2;j++)
      {
	square_kapa1_over_klapa0[i] = 1/square_kapa0_over_klapa1[i];
	vector_denominator_filter_coefficients[i][j] = denominator_filter_coefficients[i]*square_kapa1_over_klapa0[i];
      }
      
      
      // at the moment it is known that inverse of the 
      // given filter is stable, hence no numerator
      // TODO
      VectorWithOffset<float> numerator_filter_coefficients(0,1);
      numerator_filter_coefficients[0] =0;
      numerator_filter_coefficients[1] =0;
      
      for (int i =min_index;i<=max_index;i++)
	for(int j=0;j<=2;j++)
	{
	  vector_numerator_filter_coefficients[i][j] =0;	  
	}   
	
	for (int i=1;i<=num_dimensions;i++)
	{
         all_1d_array_filters[i-1] = 
//	 all_1d_array_filters[0] = 	 
	    new VectorsArrayFilter1DUsingRecursiveConvolution<float>(vector_denominator_filter_coefficients,vector_numerator_filter_coefficients);

	}
	
	
}
#endif


#if 0
template <int num_dimensions, typename elemT>
ModifiedInverseAverigingArrayFilter<num_dimensions,elemT>::
ModifiedInverseAverigingArrayFilter(Array<num_dimensions,Array<1,vector<float> > >& vector_kapa0_over_klapa1)
{   
  // default coefficients
  // TODO 
  vector<float> denominator_filter_coefficients(0,2);
  denominator_filter_coefficients[0] =0.25;
  denominator_filter_coefficients[1] =0.5;
  denominator_filter_coefficients[2] =0.25;
  
  vector<float>  denominator_coeff_mult(3);
  // all dimeniosna will have the same number of elements per
  // outter index
  int min_index = vector_kapa0_over_klapa1[0].get_min_index();
  int max_index = vector_kapa0_over_klapa1[0].get_max_index();
  
  Array<num_dimensions, VectorWithOffset< vector<float> > > square_vector_kapa0_over_klapa1(0,num_dimensions);
  Array<num_dimensions, VectorWithOffset< vector<float> > > square_vector_kapa1_over_klapa0(0,num_dimensions);
  
  // grow
  for (int i=0;i<=num_dimensions;i++)
  {
    square_vector_kapa0_over_klapa1[i].grow(min_index,max_index);
    square_vector_kapa1_over_klapa0[i].grow(min_index,max_index);
  }
  
  //VectorWithOffset<VectorWithOffset<float> > vector_denominator_filter_coefficients(min_index, max_index);
  Array<num_dimensions,VectorWithOffset< vector<float> > > array_of_vector_denominator_filter_coefficients(0, num_dimensions);
  Array<num_dimensions,VectorWithOffset< vector<float> > > array_of_vector_numerator_filter_coefficients(0,num_dimensions);
  
  // grow inner indices
  for (int i=0;i<=num_dimensions;i++)
  { 
    array_of_vector_denominator_filter_coefficients[i].grow(min_index,max_index);
    array_of_vector_numerator_filter_coefficients[i].grow(min_index,max_index);    
  }
  
  for (int i= 0;i<= num_dimensions;i++)
    for (int j= vector_kapa0_over_klapa1[i].get_min_index();i<=vector_kapa0_over_klapa1[i].get_max_index();i++)
    {
      array_of_vector_denominator_filter_coefficients[i][j].grow(0,2);
      array_of_vector_numerator_filter_coefficients[i][j].grow(0,2);
    }
    
    for ( int i = 0;i<=num_dimensions;i++)
      for (int j =min_index,j<=max_index;j++)
      {
	square_vector_kapa0_over_klapa1[i][j] = (vector_kapa0_over_klapa1[i][j])*(vector_kapa0_over_klapa1[i][j]);
      }
      
      for ( int i = 0;i<=num_dimensions;i++)
	for (int j =min_index,j<=max_index;j++)
	{
	  square_vector_kapa0_over_klapa1[i][j] = 1- square_vector_kapa0_over_klapa1[i][j];
	}
	
	for (int k =0;k<=num_dimensions;k++) 
	  for (int i =min_index;i<=max_index;i++)
	    for(int j=0;j<=2;j++)
	    {
	      array_of_vector_denominator_filter_coefficients[k][i][j] = denominator_filter_coefficients[j]* square_vector_kapa0_over_klapa1[i][j];
	      array_of_vector_denominator_filter_coefficients[k][i][j] *=-1;    
	      array_of_vector_denominator_filter_coefficients[k][i][j] +=1;
	      
	    }
	    
	    for (int k =0;k<=num_dimensions;k++) 
	      for (int i =min_index;i<=max_index;i++)
		for(int j=0;j<=2;j++)
		{
		  square_kapa1_over_klapa0[i][j] = 1/square_kapa0_over_klapa1[i][j];
		  array_of_vector_denominator_filter_coefficients[k][i][j]*=square_kapa1_over_klapa0[i][j];
		}
		
		// at the moment it is known that inverse of the 
		// given filter is stable, hence no numerator
		// TODO
		vector<float> numerator_filter_coefficients(0,1);
		numerator_filter_coefficients[0] =0;
		numerator_filter_coefficients[1] =0;
		
		for (int k =0;k<=num_dimensions;k++)
		  for (int i =min_index;i<=max_index;i++)
		    for(int j=0;j<=2;j++)
		    {
		      array_of_vector_numerator_filter_coefficients[k][i][j] =0;	  
		    }   
		    
		    for (int i=1;i<=num_dimensions;i++)
		    {
		      all_1d_array_filters[i-1] = 	 
			new VectorsArrayFilter1DUsingRecursiveConvolution<float>(array_of_vector_denominator_filter_coefficients[i],array_of_vector_numerator_filter_coefficients[i]);
		      
		    }
		    
		    
}

#endif
