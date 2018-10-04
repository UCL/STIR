

#include "stir_experimental/local_helping_functions.h"
#include "stir/IndexRange2D.h"
#include "stir/info.h"
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



START_NAMESPACE_STIR

// divide complex arrays where elements are stored as follows:
// A = array[ a(1), a(2), a(3), a(4), a(5), a(6), a(7), a(8), a(9),a(10)]
// Re[A] = even coeff
// Im[A] = odd coeff 


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

// two argument implementation


void mulitply_complex_arrays(Array<1,float>& array_nom,
			      const Array<1,float>& array_denom)
{
  
  
  int i = 1;
  int j = 2;
  while (j <=array_nom.get_max_index())
  {
    float tmp1 =(array_nom[i]*array_denom[i] - array_nom[j]*array_denom[j]);
    float tmp2 =(array_nom[j]*array_denom[i]+ array_nom[i]*array_denom[j]);

    array_nom[i] =tmp1 ;
    array_nom[j] =tmp2 ;
    i=i+2;
    j=j+2;
    
  }
}



void divide_complex_arrays( Array<1,float>& array_nom,
			     const Array<1,float>& array_denom)
{
  //assert(array_nom.get_length()== array_denom.get_length());
  //assert(out_array.get_length()== array_denom.get_length());
  //out_array.fill(0);
  
  int i = 1;
  int j = 2;
  while (j <=array_nom.get_max_index())
  {
    float denom =  (array_denom[i]*array_denom[i])+(array_denom[j]*array_denom[j]);
    float out1 = (array_nom[i]*array_denom[i] + array_nom[j]*array_denom[j])/denom;
    float out2 = (array_nom[j]*array_denom[i]-array_nom[i]*array_denom[j])/denom;

    array_nom[i] = out1;    
    array_nom[j] = out2;
    i=i+2;
    j=j+2;
    
  }
}
 


void create_kernel_3d ( Array<3,float>& kernel_3d, const VectorWithOffset < float>& kernel_1d)
{

  if (kernel_3d.get_min_index() ==kernel_3d.get_max_index())
  {
      info("In the right loop");
      Array<2,float> kernel_2d(IndexRange2D(kernel_1d.get_min_index(), kernel_1d.get_max_index(),
      kernel_1d.get_min_index(), kernel_1d.get_max_index()));
  
    for ( int j = kernel_3d[kernel_3d.get_min_index()].get_min_index(); j<=kernel_3d[kernel_3d.get_min_index()].get_max_index();j++)
      for ( int i = kernel_3d[kernel_3d.get_min_index()][j].get_min_index(); i<=kernel_3d[kernel_3d.get_min_index()][j].get_max_index();i++)
      {
	kernel_2d[j][i] = kernel_1d[j]*kernel_1d[i];
      }
      for ( int k = kernel_3d.get_min_index(); k<=kernel_3d.get_max_index();k++)
	for ( int j = kernel_3d[k].get_min_index(); j<=kernel_3d[k].get_max_index();j++)
	  for ( int i = kernel_3d[k][j].get_min_index(); i<=kernel_3d[k][j].get_max_index();i++)
	  {
	    kernel_3d[k][j][i] = kernel_2d[j][i];
	  }
        }
      else
      {
	for ( int k = kernel_3d.get_min_index(); k<=kernel_3d.get_max_index();k++)
	  for ( int j = kernel_3d[k].get_min_index(); j<=kernel_3d[k].get_max_index();j++)
	    for ( int i = kernel_3d[k][j].get_min_index(); i<=kernel_3d[k][j].get_max_index();i++)
	    {
	      kernel_3d[k][j][i] = kernel_1d[k]*kernel_1d[j]*kernel_1d[i];
	    }
      }
     
  
}

void create_kernel_2d ( Array<2, float> & kernel_2d, const VectorWithOffset < float>& kernel_1d)
{

    for ( int j = kernel_2d.get_min_index(); j<=kernel_2d.get_max_index();j++)
      for ( int i = kernel_2d[j].get_min_index(); i<=kernel_2d[j].get_max_index();i++)
      {
       kernel_2d[j][i] = kernel_1d[j]*kernel_1d[i];
      }


}

// this function padds 3d filter kernel with zeros and also makes it symmetric
void padd_filter_coefficients_3D_and_make_them_symmetric(VectorWithOffset < VectorWithOffset < VectorWithOffset < float> > > &padded_filter_coefficients_3D,
							 VectorWithOffset < VectorWithOffset < VectorWithOffset < float> > > &filter_coefficients)
							 
{
  int min_z = padded_filter_coefficients_3D.get_min_index();
  int max_z = padded_filter_coefficients_3D.get_max_index();
  int min_y = padded_filter_coefficients_3D[min_z].get_min_index();
  int max_y = padded_filter_coefficients_3D[min_z].get_max_index();
  int min_x = padded_filter_coefficients_3D[min_z][min_y].get_min_index();
  int max_x = padded_filter_coefficients_3D[min_z][min_y].get_max_index();

  int start_z = (filter_coefficients.get_max_index() -filter_coefficients.get_min_index())/2 -1;
  int start_y = (filter_coefficients[start_z].get_max_index() -filter_coefficients[start_z].get_min_index())/2 -1;
  int start_x =  (filter_coefficients[start_z][start_y].get_max_index() -filter_coefficients[start_z][start_y].get_min_index())/2 -1;
  int end_z = filter_coefficients.get_max_index();
  int end_y = filter_coefficients[end_z].get_max_index();
  int end_x = filter_coefficients[end_z][end_y].get_max_index();
   
  int size = padded_filter_coefficients_3D.get_length();
  
  for ( int k=start_z; k <= end_z;k++)
  {
    for ( int j=start_y; j <= end_y;j++)
    {
      for ( int i=start_x; i <= end_x;i++)
      {
	float tmp = filter_coefficients[k][j][i];
	padded_filter_coefficients_3D[k+min_z][j+min_y][i+min_x] = filter_coefficients[k][j][i];
        info(boost::format("%1%  ") % padded_filter_coefficients_3D[k+min_z][j+min_y][i+min_x]);
	float tmp_1 = padded_filter_coefficients_3D[k+min_z][j+min_y][i+min_x];
	//padded_filter_coefficients_3D[size-pad_k][size-pad_j][size-pad_i] = filter_coefficients[k][j][i];
      } 
    }
  }

  info(" Padded filter coefficients ");
  for ( int k = -4;k<=4;k++)
    for ( int j = -4;j<=4;j++)
      for ( int i = -4;i<=4;i++)
      {
	info(boost::format("%1%  ") % padded_filter_coefficients_3D[k][j][i]);
      }
  
  
}


// convert 3d array into 1d where the output size is (x_size*y_size*z_size)*2. the faactor 2
// is used for the imaginary part in FT.
void convert_array_3D_into_1D_array( Array<1,float>& out_array,const Array<3,float>& in_array)
{
  // check the sizes -- the outsput array should be twice the size of the input array because
  // the data is stored as multidimesional complex array with real nad imaginary part 
  assert ( out_array.get_length() == (2 *in_array.get_length()*in_array[in_array.get_min_index()].get_length()* in_array[in_array.get_min_index()][in_array.get_min_index()].get_length()));
  
#if 1
  assert(out_array.get_min_index()==1);
  assert(in_array.get_min_index()==1);
  assert(in_array[1].get_min_index()==1);
  assert(in_array[1][1].get_min_index()==1);
  
  Array<1,float> tmp_out_array(out_array.get_min_index(), out_array.get_max_index());					    

  int y_size = in_array[in_array.get_min_index()].get_length();
  int x_size = in_array[in_array.get_min_index()][in_array.get_min_index()].get_length();
  for ( int k = in_array.get_min_index(); k<=in_array.get_max_index(); k++)
    for ( int j = in_array[k].get_min_index(); j<=in_array[k].get_max_index(); j++)
      for ( int i = in_array[k][j].get_min_index(); i<=in_array[k][j].get_max_index(); i++) 
      {
	 float tmp = in_array[k][j][i];
	 float  tmp_1 = ((j-1)*y_size+i)+(k-1)*(y_size*x_size);
	 tmp_out_array[((j-1)*y_size+i)+(k-1)*(y_size*x_size)] = in_array[k][j][i];
	 
      }
 
   out_array[1] =  tmp_out_array[1];
        
  int r = 3;
  for ( int k = tmp_out_array.get_min_index()+1; k<=tmp_out_array.get_max_index()/2; k++)
      {
	out_array[r] = tmp_out_array[k];      
	r +=2;
      }
#else
  Array<1,float>::iterator iter_1d = out_array.begin();
  Array<3,float>::const_full_iterator iter_3d = in_array.begin_all();
  while(iter_1d != out_array.end())
  {
    *iter_1d++ = *iter_3d++; // real part
    *iter_1d++ = 0; // imaginary part
  }
#endif
}

// real part only -> into 3D ( complex parta already separated)
void convert_array_1D_into_3D_array( Array<3,float>& out_array,const Array<1,float>& in_array)
{
#if 1
  assert(in_array.get_min_index()==1);
  assert(out_array.get_min_index()==1);
  assert(out_array[1].get_min_index()==1);
  assert(out_array[1][1].get_min_index()==1);

 int z_size = out_array.get_length();
 int y_size = out_array[out_array.get_min_index()].get_length();
 int x_size = out_array[out_array.get_min_index()][out_array.get_min_index()].get_length();


 for ( int k = out_array.get_min_index(); k<=out_array.get_max_index(); k++)
    for ( int j = out_array[k].get_min_index(); j<=out_array[k].get_max_index(); j++)
      for ( int i = out_array[k][j].get_min_index(); i<=out_array[k][j].get_max_index(); i++)
      {
	out_array[k][j][i] = in_array[((j-1)*y_size+i)+(k-1)*(y_size*x_size)];
      }
#else
  Array<3,float>::full_iterator iter_3d = out_array.begin_all();
  Array<1,float>::const_iterator iter_1d = in_array.begin();
  while(iter_1d != in_array.end())
  {
    *iter_3d++ = *iter_1d++; // real part
    iter_1d++; // skip imaginary part
  }
#endif

}

 


// convert 2d array into 1d where the output size is (x_size*y_size)*2. the faactor 2
// is used for the imaginary part in FT.
void convert_array_2D_into_1D_array( Array<1,float>& out_array,Array<2,float>& in_array)
{
  
  Array<1,float> tmp_out_array(out_array.get_min_index(), out_array.get_max_index());					    

  int y_size = in_array.get_length();
  int x_size = in_array[in_array.get_min_index()].get_length();
  // check the sizes -- the outsput array should be twice the size of the input array because
  // the data is stored as multidimesional complex array with real nad imaginary part 
//  assert ( out_array.get_length() == (2 *in_array.get_length()*in_array[in_array_get_min_index()].get_length()* in_array[in_array_get_min_index()][in_array_get_min_index()].get_length()));
  
  for ( int j = in_array.get_min_index(); j<=in_array.get_max_index(); j++)
      for ( int i = in_array[j].get_min_index(); i<=in_array[j].get_max_index(); i++) 
      {
	 float tmp = in_array[j][i];
	 float  tmp_1 = ((j-1)*y_size+i);
	 tmp_out_array[((j-1)*y_size+i)] = in_array[j][i];
	 
      }
 
   out_array[1] =  tmp_out_array[1];
        
  int r = 3;
  for ( int k = tmp_out_array.get_min_index()+1; k<=tmp_out_array.get_max_index()/2; k++)
      {
	out_array[r] = tmp_out_array[k];      
	r +=2;
      }

}

// real part only -> into 3D ( complex parta already separated)
void convert_array_1D_into_2D_array( Array<2,float>& out_array,Array<1,float>& in_array)
{

 int y_size = out_array.get_length();
 int x_size = out_array[out_array.get_min_index()].get_length();


    for ( int j = out_array.get_min_index(); j<=out_array.get_max_index(); j++)
      for ( int i = out_array[j].get_min_index(); i<=out_array[j].get_max_index(); i++)
      {
	out_array[j][i] = in_array[((j-1)*y_size+i)];
      }
}


void precompute_filter_coefficients_for_second_apporach(VoxelsOnCartesianGrid<float>& precomputed_coefficients,
							const VoxelsOnCartesianGrid<float>& input_image,
							VoxelsOnCartesianGrid<float>& sensitivity_image,
							VoxelsOnCartesianGrid<float>& normalised_bck)
{

  for (int k=input_image.get_min_z();k<=input_image.get_max_z();k++)   
    for (int j =input_image.get_min_y();j<=input_image.get_max_y();j++)
      for (int i =input_image.get_min_x();i<=input_image.get_max_x();i++)
      {
	if ( sensitivity_image[k][j][i]> 0.00001  ||sensitivity_image[k][j][i] < -0.00001)
	precomputed_coefficients[k][j][i] = input_image[k][j][i] /sensitivity_image[k][j][i];
	precomputed_coefficients[k][j][i] = precomputed_coefficients[k][j][i] *normalised_bck[k][j][i];
      }
	



}
	      
 
END_NAMESPACE_STIR
