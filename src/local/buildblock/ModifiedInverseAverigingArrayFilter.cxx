
#include "local/tomo/ModifiedInverseAverigingArrayFilter.h"
#include "local/tomo/ArrayFilter1DUsingConvolution.h"
#include "tomo/ArrayFilter1DUsingConvolutionSymmetricKernel.h"
#include "Array.h"
#include "local/fft.h"
#include <iostream>
#include <fstream>

#ifndef TOMO_NO_NAMESPACES
using std::ios;
using std::fstream;
using std::iostream;
using std::cerr;
using std::endl;
#endif



START_NAMESPACE_TOMO

// divide complex arrays where elements are stored as follows:
// A = array[ a(1), a(2), a(3), a(4), a(5), a(6), a(7), a(8), a(9),a(10)]
// Re[A] = even coeff
// Im[A] = odd coeff 

void mulitply_complex_arrays(Array<1,float>& out_array, const Array<1,float>& array_nom,
			      const Array<1,float>& array_denom);

void divide_complex_arrays( Array<1,float>& out_array, const Array<1,float>& array_nom,
			      const Array<1,float>& array_denom);

# if 1
void divide_complex_arrays(Array<1,float>& out_array, const Array<1,float>& array_nom,
			     const Array<1,float>& array_denom)
{
  assert(array_nom.get_length()== array_denom.get_length());
  assert(out_array.get_length()== array_denom.get_length());
  out_array.fill(0);
  
  int i = 1;
  int j = 2;
  while (j <=array_nom.get_max_index())
  {
    float tmp =  (array_denom[i]*array_denom[i])+(array_denom[j]*array_denom[j]);
    out_array[i] = (array_nom[i]*array_denom[i] + array_nom[j]*array_denom[j])/tmp;
    out_array[j] =(array_nom[j]*array_denom[i]-array_nom[i]*array_denom[j])/tmp;
    i=i+2;
    j=j+2;
    
  }
}

void mulitply_complex_arrays(Array<1,float>& out_array, const Array<1,float>& array_nom,
			      const Array<1,float>& array_denom)
{
  assert(array_nom.get_length()== array_denom.get_length());
  assert(out_array.get_length()== array_denom.get_length());
  out_array.fill(0);
  
  int i = 1;
  int j = 2;
  while (j <=array_nom.get_max_index())
  {
    out_array[i] = (array_nom[i]*array_denom[i] - array_nom[j]*array_denom[j]);
    out_array[j] =(array_nom[j]*array_denom[i]+ array_nom[i]*array_denom[j]);
    i=i+2;
    j=j+2;
    
  }
}
#endif

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
:filter_coefficients(0), kapa0_over_kapa1(0)
{
  //kapa0_over_kapa1=0;
   
  for (int i=1;i<=num_dimensions;i++)
  {
    all_1d_array_filters[i-1] = 	       
	new ArrayFilter1DUsingConvolution<float>();
  }

}


// SM new 17/09/2001
#if 1
template <int num_dimensions, typename elemT>
ModifiedInverseAverigingArrayFilter<num_dimensions,elemT>::
ModifiedInverseAverigingArrayFilter(const VectorWithOffset<elemT>& filter_coefficients_v,
				    const float kapa0_over_kapa1_v)
				    :kapa0_over_kapa1(kapa0_over_kapa1_v),
				    filter_coefficients(filter_coefficients_v)
{

  
    //VectorWithOffset<float> new_filter_coefficients(filter_coefficients_v.get_min_index(),filter_coefficients_v.get_max_index());    
    
#if 1
   int size =32;  
  
  
  VectorWithOffset<float> filter_coefficients_padded(1,size);
  filter_coefficients_padded.fill(0);
  filter_coefficients_padded[1] = filter_coefficients[0];
  filter_coefficients_padded[3] = filter_coefficients[1];
  filter_coefficients_padded[size-1] = filter_coefficients[-1];

  /*for (int i = 1;i<=size;i++)
    cerr << filter_coefficients_padded[i] << "   " ;
    cerr << endl;*/


  // rescale to DC=1
  float sum =0;  
  for (int i=1;i<=size;i++)
    sum += filter_coefficients_padded[i];  
  
  for (int i=1;i<=size;i++)
    filter_coefficients_padded[i] =filter_coefficients_padded[i]/sum;

  
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
  
  //TODO CHECK

 /* cerr << "Now printing the value of kapas" << endl;
  cerr << kapa0_over_kapa1_v << endl;*/
  float sq_kapas = kapa0_over_kapa1_v; 
  VectorWithOffset<float> new_filter_coefficients;

  if ( sq_kapas > 10000)
  {
    new_filter_coefficients.grow(0,0);
    new_filter_coefficients[0]=1;
  }
  else
    if (sq_kapas!=1.F)
  {
    
      int kernel_length=0;

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
  
   // new
  for (int i=1;i<=filter_coefficients_padded.get_length()/2;i++)
  { 
    if (fabs((double) real_div[i])<= real_div[real_div.get_min_index()]*1/10) break;
    else (kernel_length)++;
  }  

  new_filter_coefficients.grow(-kernel_length,kernel_length);    
  new_filter_coefficients[0] = real_div[1];
   for (int  i = 1;i<=kernel_length;i++)
   {
     new_filter_coefficients[i]=real_div[i+1];
     new_filter_coefficients[-i]=real_div[i+1];
   }

  }
    else
   {
    new_filter_coefficients = filter_coefficients;
   }
   
   // new_filter_coefficients[0] =real_div[1];
  //  new_filter_coefficients[-1] =real_div[2];
  //  new_filter_coefficients[1] =real_div[2];
  

#endif 
   cerr << " COEFF PRINT NOW" << endl;
   for (int i=new_filter_coefficients.get_min_index();i<=new_filter_coefficients.get_max_index();i++)
    cerr << new_filter_coefficients[i] << "   ";   

  const string filename ="coeff_SA_2D_pf_new";
  shared_ptr<iostream> output = new fstream (filename.c_str(), ios::out|ios::binary);
  if (!*output)
    error("Error opening output file %s\n",filename.c_str()); 

  // now rescaled the calculated coefficients to DC gain 1
  
  //float sum_new_coefficients =0.F;  
  //for (int i=-1;i<=1;i++)
   // sum_new_coefficients += new_filter_coefficients[i];  
  
 // for (int i=-1;i<=1;i++)
   // new_filter_coefficients[i] /=sum_new_coefficients;  

  *output << "coeff" << endl;
  *output << endl;  
  for (int i=new_filter_coefficients.get_min_index();i<=new_filter_coefficients.get_max_index();i++)
  *output << new_filter_coefficients[i] << "   ";
  *output << endl;

  /* cerr << " COEFF" << endl;
  for (int i=0;i<=2;i++)
  cerr << filter_coefficients[i] << "   ";*/
  
  // to do only filtering in 2d -> 
  // z-direction is for 0 index

 for (int i=2;i<=num_dimensions;i++)
  {
    all_1d_array_filters[i-1] = 
        new ArrayFilter1DUsingConvolution<float>(new_filter_coefficients);
  }
  
  all_1d_array_filters[0] = 	 
      new ArrayFilter1DUsingConvolution<float>();
  
}


#endif

// SM new
#if 0
template <int num_dimensions, typename elemT>
ModifiedInverseAverigingArrayFilter<num_dimensions,elemT>::
ModifiedInverseAverigingArrayFilter(const VectorWithOffset<elemT>& filter_coefficients_v,
				    const float kapa0_over_kapa1_v)
				    :kapa0_over_kapa1(kapa0_over_kapa1_v),
				    filter_coefficients(filter_coefficients_v)
{

  VectorWithOffset<float> new_filter_coefficients(-1,1);    
    //VectorWithOffset<float> new_filter_coefficients(filter_coefficients_v.get_min_index(),filter_coefficients_v.get_max_index());    
    
#if 1
  //float kapa1_over_kapa0= 1/kapa0_over_kapa1_v;
  int size =32;  
  
  VectorWithOffset<float> filter_coefficients_padded(1,size);
  filter_coefficients_padded.fill(0);
  filter_coefficients_padded[1] = filter_coefficients[0];
  filter_coefficients_padded[3] = filter_coefficients[1];
  filter_coefficients_padded[size-1] = filter_coefficients[-1];

  /*for (int i = 1;i<=size;i++)
    cerr << filter_coefficients_padded[i] << "   " ;
    cerr << endl;*/

#if 0
  filter_coefficients_padded[5] = filter_coefficients[0];
  filter_coefficients_padded[7] = filter_coefficients[1];
  filter_coefficients_padded[9] = filter_coefficients[2];

  // new
 // filter_coefficients_padded[11] = 1;
 // filter_coefficients_padded[13] = filter_coefficients[2];
 // filter_coefficients_padded[15] = filter_coefficients[1];
 // filter_coefficients_padded[17] = filter_coefficients[0];  
  
  
  filter_coefficients_padded[11] = filter_coefficients[2];
  filter_coefficients_padded[13] = filter_coefficients[1];
  filter_coefficients_padded[15] = filter_coefficients[0];  
#endif

  // rescale to DC=1
  float sum =0;  
  for (int i=1;i<=size;i++)
    sum += filter_coefficients_padded[i];  
  
  for (int i=1;i<=size;i++)
    filter_coefficients_padded[i] =filter_coefficients_padded[i]/sum;

   /*if (kapa0_over_kapa1_v==1)
  {
    new_filter_coefficients[0] = filter_coefficients_v[0];
    new_filter_coefficients[1] = filter_coefficients_v[1];
    new_filter_coefficients[2] = filter_coefficients_v[2];
  }
  else
  { */
  
  Array<1,float> fft_filter(1,filter_coefficients_padded.get_length());
  Array<1,float> fft_1(1,filter_coefficients_padded.get_length());  

  Array<1,float> fft_filter_denom(1,filter_coefficients_padded.get_length());
  Array<1,float> fft_filter_denom_1(1,filter_coefficients_padded.get_length());


  Array<1,float> fft_filter_num(1,filter_coefficients_padded.get_length());
  Array<1,float> div(1,filter_coefficients_padded.get_length());
  
  fft_filter_denom.fill(0);
  fft_filter_num.fill(0);
  div.fill(0);
  
  // new - HERE
  //fft_1[9] =1;
    fft_1[1] =1;
  // new
  //fft_1[11] =1;
  

  for (int i =1;i<=size;i++)
    fft_filter[i] = filter_coefficients_padded[i]; 
  
  //TODO CHECK

 /* cerr << "Now printing the value of kapas" << endl;
  cerr << kapa0_over_kapa1_v << endl;*/
  float sq_kapas = kapa0_over_kapa1_v; //kapa1_over_kapa0;
  //cerr << "CHECKING " << endl;
   // cerr << sq_kapas << endl;

  
  float inverse_sq_kapas;
  if (fabs((double)sq_kapas ) >0.000000000001)
    inverse_sq_kapas = 1/sq_kapas;
  else 
    inverse_sq_kapas = 0;
  
  fft_filter_denom =fft_filter*(sq_kapas-1) + fft_1;
  fft_filter_denom *= inverse_sq_kapas;
  
  four1(fft_filter_denom,fft_filter_denom.get_length()/2,1);
  // new optimised
  /*FFT_routines fft_filter_new;
  FFT_routines fft_unity_new;
  fft_filter_new.find_fft_filter(fft_filter);
  fft_filter_new.find_fft_unity(fft_1);*/

  four1(fft_1,fft_1.get_length()/2,1);
  four1(fft_filter,fft_filter.get_length()/2,1);
  
  
  // to check the outputs make the fft consistant with mathematica
  // divide 1/sqrt(size/2)
  for (int i=1; i<=size; i++)
  {
    fft_filter[i] =fft_filter[i]/sqrt(size/2);
    fft_1[i] =fft_1[i]/sqrt(size/2);
    fft_filter_denom[i] = fft_filter_denom[i]/sqrt(size/2);
  }
  /* cerr << " fft_denom" << endl;  
   for (int i=1;i<=size;i++)
    cerr << fft_filter_denom[i] << "   ";

   cerr << endl;*/
  
  mulitply_complex_arrays(fft_filter_num,fft_filter,fft_1);
  divide_complex_arrays(div,fft_filter_num,fft_filter_denom); 

  
  four1(div,div.get_length()/2,-1);
  
  for (int i = div.get_min_index();i<=div.get_max_index();i++)
  {
    div[i] = div[i]/sqrt(div.get_length()/2);
  }
  
  Array<1,float> real_div(1,filter_coefficients_padded.get_length()/2);
  real_div[1] = div[1];
  for (int i=1;i<=(size/2)-1;i++)
    real_div[i+1] = div[2*i+1];
  
    /* cerr << "div" <<endl;
    cerr << div[1] <<"   "; 
    for (int i=1;i<size/2;i++)
    cerr << div[2*i+1] << "    ";
  cerr << endl;*/
  
  /*cerr << "REAL coeff" << endl;
  for (int i=1;i<=real_div.get_max_index();i++)
    cerr << real_div[i] << "    ";
    cerr << endl;*/


  // sm 12/09/2001
    /*new_filter_coefficients[1] =real_div[1];
    new_filter_coefficients[0] =real_div[2];
    new_filter_coefficients[2] =real_div[2];*/


    new_filter_coefficients[0] =real_div[1];
    new_filter_coefficients[-1] =real_div[2];
    new_filter_coefficients[1] =real_div[2];
  

#endif 
   //cerr << " COEFF PRINT NOW" << endl;
   //for (int i=0;i<=2;i++)
   // cerr << new_filter_coefficients[i] << "   ";
   //cerr << endl;

  const string filename ="coeff_RT_2D_TEST";
  shared_ptr<iostream> output = new fstream (filename.c_str(), ios::out|ios::binary|ios::app|ios::ate);
  if (!*output)
    error("Error opening output file %s\n",filename.c_str()); 

  // now rescaled the calculated coefficients to DC gain 1
  
  float sum_new_coefficients =0.F;  
  for (int i=-1;i<=1;i++)
    sum_new_coefficients += new_filter_coefficients[i];  
  
  for (int i=-1;i<=1;i++)
    new_filter_coefficients[i] /=sum_new_coefficients;  

  *output << "coeff" << endl;
  *output << endl;  
  for (int i=-1;i<=1;i++)
  *output << new_filter_coefficients[i] << "   ";
  *output << endl;

  /* cerr << " COEFF" << endl;
  for (int i=0;i<=2;i++)
  cerr << filter_coefficients[i] << "   ";*/
  
  // to do only filtering in 2d -> 
  // z-direction is for 0 index

 for (int i=2;i<=num_dimensions;i++)
  {
    all_1d_array_filters[i-1] = 
        new ArrayFilter1DUsingConvolution<float>(new_filter_coefficients);
  }
  
  all_1d_array_filters[0] = 	 
      new ArrayFilter1DUsingConvolution<float>();
  
}

// SM old 
#if 0
template <int num_dimensions, typename elemT>
ModifiedInverseAverigingArrayFilter<num_dimensions,elemT>::
ModifiedInverseAverigingArrayFilter(const VectorWithOffset<elemT>& filter_coefficients_v,
				    const float kapa0_over_kapa1_v)
				    :kapa0_over_kapa1(kapa0_over_kapa1_v),
				    filter_coefficients(filter_coefficients_v)
{

  //VectorWithOffset<float> new_filter_coefficients(0,2);    
    VectorWithOffset<float> new_filter_coefficients(filter_coefficients_v.get_min_index(),filter_coefficients_v.get_max_index());    
    
#if 1
  //float kapa1_over_kapa0= 1/kapa0_over_kapa1_v;
  int size =32;  
  
  VectorWithOffset<float> filter_coefficients_padded(1,size);
  filter_coefficients_padded.fill(0);
  filter_coefficients_padded[1] = filter_coefficients[1];
  filter_coefficients_padded[3] = filter_coefficients[2];
  filter_coefficients_padded[size-1] = filter_coefficients[0];

 // for (int i = 1;i<=size;i++)
   // cerr << filter_coefficients_padded[i] << "   " ;
   // cerr << endl;

#if 0
  filter_coefficients_padded[5] = filter_coefficients[0];
  filter_coefficients_padded[7] = filter_coefficients[1];
  filter_coefficients_padded[9] = filter_coefficients[2];

  // new
 // filter_coefficients_padded[11] = 1;
 // filter_coefficients_padded[13] = filter_coefficients[2];
 // filter_coefficients_padded[15] = filter_coefficients[1];
 // filter_coefficients_padded[17] = filter_coefficients[0];  
  
  
  filter_coefficients_padded[11] = filter_coefficients[2];
  filter_coefficients_padded[13] = filter_coefficients[1];
  filter_coefficients_padded[15] = filter_coefficients[0];  
#endif

  // rescale to DC=1
  float sum =0;  
  for (int i=1;i<=size;i++)
    sum += filter_coefficients_padded[i];  
  
  for (int i=1;i<=size;i++)
    filter_coefficients_padded[i] =filter_coefficients_padded[i]/sum;

   /*if (kapa0_over_kapa1_v==1)
  {
    new_filter_coefficients[0] = filter_coefficients_v[0];
    new_filter_coefficients[1] = filter_coefficients_v[1];
    new_filter_coefficients[2] = filter_coefficients_v[2];
  }
  else
  { */
  
  Array<1,float> fft_filter(1,filter_coefficients_padded.get_length());
  Array<1,float> fft_1(1,filter_coefficients_padded.get_length());  

  Array<1,float> fft_filter_denom(1,filter_coefficients_padded.get_length());
  Array<1,float> fft_filter_denom_1(1,filter_coefficients_padded.get_length());


  Array<1,float> fft_filter_num(1,filter_coefficients_padded.get_length());
  Array<1,float> div(1,filter_coefficients_padded.get_length());
  
  fft_filter_denom.fill(0);
  fft_filter_num.fill(0);
  div.fill(0);
  
  // new - HERE
  //fft_1[9] =1;
    fft_1[1] =1;
  // new
  //fft_1[11] =1;
  

  for (int i =1;i<=size;i++)
    fft_filter[i] = filter_coefficients_padded[i]; 
  
  //TODO CHECK

 /* cerr << "Now printing the value of kapas" << endl;
  cerr << kapa0_over_kapa1_v << endl;*/
  float sq_kapas = kapa0_over_kapa1_v; //kapa1_over_kapa0;
  //cerr << "CHECKING " << endl;
   // cerr << sq_kapas << endl;

  
  float inverse_sq_kapas;
  if (fabs((double)sq_kapas ) >0.000000000001)
    inverse_sq_kapas = 1/sq_kapas;
  else 
    inverse_sq_kapas = 0;
  
  fft_filter_denom =fft_filter*(sq_kapas-1) + fft_1;
  fft_filter_denom *= inverse_sq_kapas;
  
  four1(fft_filter_denom,fft_filter_denom.get_length()/2,1);
  // new optimised
  /*FFT_routines fft_filter_new;
  FFT_routines fft_unity_new;
  fft_filter_new.find_fft_filter(fft_filter);
  fft_filter_new.find_fft_unity(fft_1);*/

  four1(fft_1,fft_1.get_length()/2,1);
  four1(fft_filter,fft_filter.get_length()/2,1);
  
  
  // to check the outputs make the fft consistant with mathematica
  // divide 1/sqrt(size/2)
  for (int i=1; i<=size; i++)
  {
    fft_filter[i] =fft_filter[i]/sqrt(size/2);
    fft_1[i] =fft_1[i]/sqrt(size/2);
    fft_filter_denom[i] = fft_filter_denom[i]/sqrt(size/2);
  }
  /* cerr << " fft_denom" << endl;  
   for (int i=1;i<=size;i++)
    cerr << fft_filter_denom[i] << "   ";

   cerr << endl;*/
  
  mulitply_complex_arrays(fft_filter_num,fft_filter,fft_1);
  divide_complex_arrays(div,fft_filter_num,fft_filter_denom); 

  
  four1(div,div.get_length()/2,-1);
  
  for (int i = div.get_min_index();i<=div.get_max_index();i++)
  {
    div[i] = div[i]/sqrt(div.get_length()/2);
  }
  
  Array<1,float> real_div(1,filter_coefficients_padded.get_length()/2);
  real_div[1] = div[1];
  for (int i=1;i<=(size/2)-1;i++)
    real_div[i+1] = div[2*i+1];
  
    /* cerr << "div" <<endl;
    cerr << div[1] <<"   "; 
    for (int i=1;i<size/2;i++)
    cerr << div[2*i+1] << "    ";
  cerr << endl;*/
  
  /*cerr << "REAL coeff" << endl;
  for (int i=1;i<=real_div.get_max_index();i++)
    cerr << real_div[i] << "    ";
    cerr << endl;*/

  // try only tree coefff!!!
#if 0
  
  //VectorWithOffset<float> new_filter_coefficients_long(1,size);
  int kernel_length_l=0;
  // find out a size of the new filter coefficients

  // new
  for (int i=4;i>=0;i--)
  // for (int i=5;i>0;i--)

  { 
    //cerr << div[i] << "    " << div[7];
    //if (fabs((double) real_div[i])<=(0.1)*real_div[5]) break;
    if (fabs((double) real_div[i])<= real_div[5]/3) break;

     //new
    // if (fabs((double) real_div[i])<=(0.01)*real_div[6]) break;

    else (kernel_length_l)++;
  }  
  int kernel_length_r=0;  
  for (int i=5;i<=real_div.get_max_index();i++)
  //for (int i=6;i<=real_div.get_max_index();i++)

  { 
    //if (fabs((double) real_div[i])<=(0.1)*real_div[5]) break;
     if (fabs((double) real_div[i])<= real_div[5]/3) break;
    // new
    //if (fabs((double) real_div[i])<=(0.1)*real_div[6]) break;

    else (kernel_length_r)++;
  }
  
  cerr << "kernel_L "<< kernel_length_l << endl;
  cerr << "kernel_H "<< kernel_length_r << endl; 
  
  
  int sum_lenghts = kernel_length_l + kernel_length_r;
  VectorWithOffset<float> new_filter_coefficients(1,sum_lenghts);

  // new 
  // rescale to DC=1
 /* float sum1 =0;  
  for (int i=1;i<=sum_lenghts;i++)
    sum1 += new_filter_coefficients[i];  

 for (int i=1;i<=sum_lenghts;i++)
    new_filter_coefficients[i] =new_filter_coefficients[i]/sum1;   */
 

  new_filter_coefficients[kernel_length_l] = real_div[5];
  // new
  //new_filter_coefficients[kernel_length_l] = real_div[6];
  int j=0; 
  for (int i=kernel_length_l-1;i>0;i--)
  { j = j+1;
  new_filter_coefficients[i] = real_div[5-j];}
  //new 
  //new_filter_coefficients[i] = real_div[6-j];}
  
  j=0; 
  for (int i=kernel_length_l+1;i<=sum_lenghts;i++)
  { j = j+1;
  new_filter_coefficients[i] = real_div[5+j];}
  // new
  //new_filter_coefficients[i] = real_div[6+j];}
  
#endif

  // sm 12/09/2001
    /*new_filter_coefficients[1] =real_div[1];
    new_filter_coefficients[0] =real_div[2];
    new_filter_coefficients[2] =real_div[2];*/


    new_filter_coefficients[2] =real_div[1];
    new_filter_coefficients[1] =real_div[2];
    new_filter_coefficients[3] =real_div[2];




 
  //for (int i=0;i<=2;i++)
    //new_filter_coefficients[i] = real_div[];
    // for fft_1[7] 
    //new_filter_coefficients[i] = real_div[i+3];

    //new_filter_coefficients[i] = real_div[i+4];

  

#endif 
   //cerr << " COEFF PRINT NOW" << endl;
   //for (int i=0;i<=2;i++)
   // cerr << new_filter_coefficients[i] << "   ";
   //cerr << endl;

  const string filename ="coeff_RT_2D_w";
  shared_ptr<iostream> output = new fstream (filename.c_str(), ios::out|ios::binary|ios::app|ios::ate);
  if (!*output)
    error("Error opening output file %s\n",filename.c_str()); 

  // now rescaled the calculated coefficients to DC gain 1
  
  float sum_new_coefficients =0.F;  
  for (int i=1;i<=3;i++)
    sum_new_coefficients += new_filter_coefficients[i];  
  
  for (int i=1;i<=3;i++)
    new_filter_coefficients[i] /=sum_new_coefficients;  

  *output << "coeff" << endl;
  *output << endl;  
  for (int i=-1;i<=1;i++)
  *output << new_filter_coefficients[i] << "   ";
  *output << endl;

  /* cerr << " COEFF" << endl;
  for (int i=0;i<=2;i++)
  cerr << filter_coefficients[i] << "   ";*/
  
  // to do only filtering in 2d -> 
  // z-direction is for 0 index

 for (int i=2;i<=num_dimensions;i++)
  {
    all_1d_array_filters[i-1] = 
        new ArrayFilter1DUsingConvolution<float>(new_filter_coefficients);
  }
  
  all_1d_array_filters[0] = 	 
      new ArrayFilter1DUsingConvolution<float>();
  
}

#endif

#endif

#if 0
template <int num_dimensions, typename elemT>
ModifiedInverseAverigingArrayFilter<num_dimensions,elemT>::
ModifiedInverseAverigingArrayFilter(const VectorWithOffset<elemT>& filter_coefficients_v,
				    const float kapa0_over_kapa1_v)
:kapa0_over_kapa1(kapa0_over_kapa1_v),
 filter_coefficients(filter_coefficients_v)
{
  //float kapa1_over_kapa0= 1/kapa0_over_kapa1_v;
  int size =32;  
  // TODO
  //VectorWithOffset<float> new_filter_coefficients(filter_coefficients.get_min_index(),filter_coefficients.get_max_index()+2);
  
  // make it periodic and padd it with zeros
  // TODO take into account the fact that size can be !=3
  VectorWithOffset<float> filter_coefficients_padded(1,size);
  filter_coefficients_padded.fill(0);
  filter_coefficients_padded[5] = filter_coefficients[0];
  filter_coefficients_padded[7] = filter_coefficients[1];
  filter_coefficients_padded[9] = filter_coefficients[2];
  
  filter_coefficients_padded[11] = filter_coefficients[2];
  filter_coefficients_padded[13] = filter_coefficients[1];
  filter_coefficients_padded[15] = filter_coefficients[0];
  
 
  // rescale to DC=1
  float sum =0;  
  for (int i=1;i<=size;i++)
   sum += filter_coefficients_padded[i];  
  
  for (int i=1;i<=size;i++)
   filter_coefficients_padded[i] =filter_coefficients_padded[i]/sum;  
  
  Array<1,float> fft_filter(1,filter_coefficients_padded.get_length());
  Array<1,float> fft_1(1,filter_coefficients_padded.get_length());   
  Array<1,float> fft_filter_denom(1,filter_coefficients_padded.get_length());
  Array<1,float> fft_filter_num(1,filter_coefficients_padded.get_length());
  Array<1,float> div(1,filter_coefficients_padded.get_length());
  
  fft_filter_denom.fill(0);
  fft_filter_num.fill(0);
  div.fill(0);
  
  fft_1[9] =1;
  for (int i =1;i<=size;i++)
    fft_filter[i] = filter_coefficients_padded[i]; 
 
  //TODO CHECK
  float sq_kapas = kapa0_over_kapa1_v; //kapa1_over_kapa0;
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
    fft_filter[i] =fft_filter[i]/sqrt(size/2);
    fft_1[i] =fft_1[i]/sqrt(size/2);
    fft_filter_denom[i] = fft_filter_denom[i]/sqrt(size/2);
  }

  mulitply_complex_arrays(fft_filter_num,fft_filter,fft_1);  
  divide_complex_arrays( div,fft_filter_num,fft_filter_denom);

  four1(div,div.get_length()/2,-1);

  for (int i = div.get_min_index();i<=div.get_max_index();i++)
  {
    div[i] = div[i]/sqrt(div.get_length()/2);
  }
  
  Array<1,float> real_div(1,filter_coefficients_padded.get_length()/2);
  real_div[1] = div[1];
  for (int i=1;i<=size/2-1;i++)
    real_div[i+1] = div[2*i+1];
    
 /* cerr << "div" <<endl;
  cerr << div[1] <<"   "; 
  for (int i=1;i<size/2;i++)
    cerr << div[2*i+1] << "    ";
  cerr << endl;*/

  cerr << "REAL COEFF" << endl;
  for (int i=1;i<=real_div.get_max_index();i++)
    cerr << real_div[i] << "    ";
  cerr << endl;


  //VectorWithOffset<float> new_filter_coefficients_long(1,size);
  int kernel_length_l=0;
  // find out a size of the new filter coefficients
  for (int i=4;i>0;i--)
  { 
   //cerr << div[i] << "    " << div[7];
    if (fabs((double) real_div[i])<=(0.01)*real_div[5]) break;
    else (kernel_length_l)++;
  }  
  int kernel_length_r=0;  
  for (int i=5;i<=real_div.get_max_index();i++)
  { 
    if (fabs((double) real_div[i])<=(0.01)*real_div[5]) break;
    else (kernel_length_r)++;
  }

 // cerr << "kernel_L "<< kernel_length_l << endl;
 // cerr << "kernel_H "<< kernel_length_r << endl;
  
   int sum_lenghts = kernel_length_l + kernel_length_r;
   VectorWithOffset<float> new_filter_coefficients(1,sum_lenghts);
   new_filter_coefficients[kernel_length_l] = real_div[5];
   int j=0; 
    for (int i=kernel_length_l-1;i>0;i--)
    { j = j+1;
    new_filter_coefficients[i] = real_div[5-j];}

    j=0; 
    for (int i=kernel_length_l+1;i<=sum_lenghts;i++)
    { j = j+1;
    new_filter_coefficients[i] = real_div[5+j];}


  /*for (int i = 1; i<=size;i++)
  {
    new_filter_coefficients_long[i] = div[i];
  }
   // TODO the size of kernel can be longer than the intial size
  new_filter_coefficients[0] = div[3];
  new_filter_coefficients[1] = div[5];
  new_filter_coefficients[2] = div[7];
  new_filter_coefficients[3] = div[9];
  new_filter_coefficients[4] = div[11];*/
  
  /*cerr << " COEFF" << endl;
  for (int i=1;i<=sum_lenghts;i++)
    cerr << new_filter_coefficients[i] << "   ";*/

  
  for (int i=1;i<=num_dimensions;i++)
  {
    all_1d_array_filters[i-1] = 	 
      new ArrayFilter1DUsingConvolution<float>(new_filter_coefficients);
  }
  
}

#endif

template ModifiedInverseAverigingArrayFilter<3, float>;

END_NAMESPACE_TOMO
























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
