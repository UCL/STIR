//
// $Id$
//
/*!

  \file
  \ingroup buildblock
  \brief Implementations for class ArrayFilter3DUsingConvolution

  \author Kris Thielemans
  \author Sanida Mustafovic

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2001- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/

#include "local/stir/ArrayFilter3DUsingConvolution.h"
#include "stir/IndexRange.h"
#include "stir/VectorWithOffset.h"
#include "stir/Array.h"
#include "stir/IndexRange3D.h"

START_NAMESPACE_STIR

template <typename elemT>
ArrayFilter3DUsingConvolution<elemT>::
ArrayFilter3DUsingConvolution()
: filter_coefficients()
{
  
}

template <typename elemT>
ArrayFilter3DUsingConvolution<elemT>::
ArrayFilter3DUsingConvolution(const Array <3, float> &filter_coefficients_v)
: filter_coefficients(filter_coefficients_v)
{
  // TODO: remove 0 elements at the outside
}


template <typename elemT>
bool 
ArrayFilter3DUsingConvolution<elemT>::
is_trivial() const
{
  return
    filter_coefficients.get_length() == 0 ||
    (filter_coefficients.get_length()==1 && filter_coefficients.get_min_index()==0 &&
     filter_coefficients[0][0][0] == 1);
}


template <typename elemT>
Succeeded 
ArrayFilter3DUsingConvolution<elemT>::
get_influencing_indices(IndexRange<1>& influencing_index_range, 
                        const IndexRange<1>& input_index_range) const
{
  influencing_index_range = 
    (filter_coefficients.get_length() == 0)
    ? input_index_range
    : IndexRange<1>(input_index_range.get_min_index() - filter_coefficients.get_max_index(),
                    input_index_range.get_max_index() - filter_coefficients.get_min_index());
  return Succeeded::yes;
}

template <typename elemT>
Succeeded 
ArrayFilter3DUsingConvolution<elemT>:: 
get_influenced_indices(IndexRange<1>& influenced_index_range, 
                       const IndexRange<1>& output_index_range) const
{
  influenced_index_range = 
    (filter_coefficients.get_length() == 0)
    ? output_index_range
    : IndexRange<1>(output_index_range.get_min_index() + filter_coefficients.get_min_index(),
                    output_index_range.get_max_index() + filter_coefficients.get_max_index());
  return Succeeded::yes;
}

#if 1
template <typename elemT>
void
ArrayFilter3DUsingConvolution<elemT>::
do_it(Array<3,elemT>& out_array, const Array<3,elemT>& in_array) const
{
  const int in_min_z = in_array.get_min_index();
  const int in_max_z = in_array.get_max_index();
  const int in_min_y = in_array[in_min_z].get_min_index();
  const int in_max_y = in_array[in_min_z].get_max_index();
  const int in_min_x = in_array[in_min_z][in_min_y].get_min_index();
  const int in_max_x = in_array[in_min_z][in_min_y].get_max_index();
  
  
  const int out_min_z = out_array.get_min_index();
  const int out_max_z = out_array.get_max_index();
  const int out_min_y = out_array[out_min_z].get_min_index();
  const int out_max_y = out_array[out_min_z].get_max_index();
  const int out_min_x = out_array[out_min_z][out_min_y].get_min_index();
  const int out_max_x = out_array[out_min_z][out_min_y].get_max_index();
  

  
  if (is_trivial())
  {    
    for (int z=out_min_z; z<=out_max_z; z++) 
      for (int y=out_min_y; y<=out_max_y; y++) 
	for (int x=out_min_x; x<=out_max_x; x++) 
	  
	{
	  out_array[z][y][x] = ((z>=in_min_z && z <= in_max_z ) && (y>=in_min_y && y <= in_max_y ) && 
	    (x>=in_min_x && x <= in_max_x ) ? in_array[z][y][x] : 0);   
	}
	return;
  }
  const int k_min = filter_coefficients.get_min_index();
  const int k_max = filter_coefficients.get_max_index();
  const int j_min = filter_coefficients[k_min].get_min_index();
  const int j_max = filter_coefficients[k_min].get_max_index();
  const int i_min = filter_coefficients[k_min][j_min].get_min_index();
  const int i_max = filter_coefficients[k_min][j_min].get_max_index();
  
  Array <3,float> filter_tmp (IndexRange3D(0,k_max-1,0,j_max-1,0,i_max-1));

  for ( int k =k_min; k<=k_max; k++)
    for ( int j =j_min; j<=j_max; j++)
      for ( int i =i_min; i<=i_max; i++)
      {
	filter_tmp[k-1][j-1][i-1]  = filter_coefficients[k][j][i];

      }
  
  const int k_min_tmp = filter_tmp.get_min_index();
  const int k_max_tmp = filter_tmp.get_max_index();
  const int j_min_tmp = filter_tmp[k_min].get_min_index();
  const int j_max_tmp = filter_tmp[k_min].get_max_index();
  const int i_min_tmp = filter_tmp[k_min][j_min].get_min_index();
  const int i_max_tmp = filter_tmp[k_min][j_min].get_max_index();

 
  for (int z=out_min_z; z<=out_max_z; z++) 
    for (int y=out_min_y; y<=out_max_y; y++) 
      for (int x=out_min_x; x<=out_max_x; x++) 
      {
	out_array[z][y][x] = 0;
	
	for (int k=max(k_min_tmp, z-in_max_z); k<=min(k_max_tmp, z-in_min_z); k++) 
	  for (int j=max(j_min_tmp, y-in_max_y); j<=min(j_max_tmp, y-in_min_y); j++)  
	    for (int i=max(i_min_tmp, x-in_max_x); i<=min(k_max_tmp, x-in_min_x); i++) 
	      
	      out_array[z][y][x] += filter_tmp[k][j][i]*in_array[z-k][y-j][x-i];   
      }
      
      
}

#endif


#if 0
template <typename elemT>
void
ArrayFilter3DUsingConvolution<elemT>::
do_it(Array<3,elemT>& out_array, const Array<3,elemT>& in_array) const
{
  const int in_min_z = in_array.get_min_index();
  const int in_max_z = in_array.get_max_index();
  const int in_min_y = in_array[in_min_z].get_min_index();
  const int in_max_y = in_array[in_min_z].get_max_index();
  const int in_min_x = in_array[in_min_z][in_min_y].get_min_index();
  const int in_max_x = in_array[in_min_z][in_min_y].get_max_index();
  
  
  const int out_min_z = out_array.get_min_index();
  const int out_max_z = out_array.get_max_index();
  const int out_min_y = out_array[out_min_z].get_min_index();
  const int out_max_y = out_array[out_min_z].get_max_index();
  const int out_min_x = out_array[out_min_z][out_min_y].get_min_index();
  const int out_max_x = out_array[out_min_z][out_min_y].get_max_index();
  

  
  if (is_trivial())
  {    
    for (int z=out_min_z; z<=out_max_z; z++) 
      for (int y=out_min_y; y<=out_max_y; y++) 
	for (int x=out_min_x; x<=out_max_x; x++) 
	  
	{
	  out_array[z][y][x] = ((z>=in_min_z && z <= in_max_z ) && (y>=in_min_y && y <= in_max_y ) && 
	    (x>=in_min_x && x <= in_max_x ) ? in_array[z][y][x] : 0);   
	}
	return;
  }
  const int k_min = filter_coefficients.get_min_index();
  const int k_max = filter_coefficients.get_max_index();
  const int j_min = filter_coefficients[k_min].get_min_index();
  const int j_max = filter_coefficients[k_min].get_max_index();
  const int i_min = filter_coefficients[k_min][j_min].get_min_index();
  const int i_max = filter_coefficients[k_min][j_min].get_max_index();
  
  
  for (int z=out_min_z; z<=out_max_z; z++) 
    for (int y=out_min_y; y<=out_max_y; y++) 
      for (int x=out_min_x; x<=out_max_x; x++) 
      {
	out_array[z][y][x] = 0;
	
	for (int k=max(k_min, z-in_max_z); k<=min(k_max, z-in_min_z); k++) 
	  for (int j=max(j_min, y-in_max_y); j<=min(j_max, y-in_min_y); j++)  
	    for (int i=max(i_min, x-in_max_x); i<=min(k_max, x-in_min_x); i++) 
	      
	      out_array[z][y][x] += filter_coefficients[k][j][i]*in_array[z-k][y-j][x-i];   
      }
      
      
}

#endif

// instantiation

template ArrayFilter3DUsingConvolution<float>;

END_NAMESPACE_STIR

