//
// $Id$: $Date$
//

#include "pet_common.h"
#include "interpolate.h"

/* 
  Implementation details:

  Because this implementation works for arbitrary (numeric) types T, it
  is slightly more complicated than would be necessary for (say) floats.
  In particular, 
  - we do our best to avoid creating temporary objects of type T
  - we zero values by using multiplication with 0 
  (actually we use T::operator*=(0)). This is to allow the case where
  T::operator=(int) does not exist (in particular, in our higher 
  dimensional arrays).


  History:
  - first version by Kris Thielemans with suggestions by Alexey Zverovich.
    (loosely based on a 1D version by Claire Labbe)
*/


template <typename T>
void
overlap_interpolate(VectorWithOffset<T>& out_data, 
		    const VectorWithOffset<T>& in_data,
		    const float zoom, 
		    const float offset, 
		    const bool assign_rest_with_zeroes)
{ 
  assert(zoom>0);
  
  // First check trivial case
  if (zoom==1.F && offset==0.F && 
    in_data.get_min_index()==out_data.get_min_index() &&
    in_data.get_max_index()==out_data.get_max_index())
  {
    out_data = in_data;
    return;
  }
  
  if(zoom>=1)
  {
    // Shrinking to a smaller bin size    
    
    // start at the first 'in' bin that overlaps the 'out' data.
    // compute by checking by comparing its position with
    // the position of the left edge of the first 'out' bin:
    // left_edge = (out_data.get_min_index() - .5)/zoom + offset
    // x1 -.5 <= left_edge < x1+.5
    int x2 = out_data.get_min_index();
    int x1 = (int)floor((x2 - .5)/zoom + offset + .5);
    
    // the next variable holds the difference between the coordinates
    // of the right edges of the 'in' bin and the 'out' bin, computed
    // in a coordinate system where the 'out' voxels are unit distance apart
    double diff_between_right_edges = 
      zoom*(x1-offset+.5) - (x2 + .5);
    
    for(; 
        x2 <= out_data.get_max_index(); 
        x2++, diff_between_right_edges--)
    {
      
      if(x1> in_data.get_max_index())
      {
	// just fill out_data with 0, 
	// no need to check/update diff_between_right_edges 
	if (assign_rest_with_zeroes) 
	  out_data[x2] *= 0;
	
	continue;
      }
      
      assert(diff_between_right_edges<= zoom/*+epsilon*/);
      assert(diff_between_right_edges>= -1/*epsilon*/);
      
      if (diff_between_right_edges >= 0)
      {
	if(x1 >= in_data.get_min_index()) 
	{
	  out_data[x2] = in_data[x1];
	  out_data[x2] /= zoom;
	}
	else 
	{
	  if (assign_rest_with_zeroes) 
	    out_data[x2] *= 0;
	}
      }
      else
      {	
        /*	
          Set out_data[x2] according to
      
          T V1; // bin value at x1
          T V2; // bin value at x1+1	
          out_data[x2] = (V1+diff_between_right_edges*(V1-V2))/zoom;
       
          The lines below are more complicated because
	  - testing if x1, x1+1 are inside the range of in_data
	  - everything is done without creating temporary objects of type T
        */
	if(x1 >= in_data.get_min_index()) 
	{
	  out_data[x2] = in_data[x1]; 
	  out_data[x2] *= 1/diff_between_right_edges + 1;
	}
	else 
	{
	  if (assign_rest_with_zeroes) 	
	    out_data[x2] *= 0;	  
	}	
	if(x1+1 <= in_data.get_max_index() && x1+1>=in_data.get_min_index()) 
	{
	  out_data[x2] -= in_data[x1+1];
	}
	
	out_data[x2] *= diff_between_right_edges/zoom;
	
	// advance 'in' bin
	x1++;
	diff_between_right_edges += zoom;
      }
    }// End of for x2
    
  }
  else
  { 
    // case zoom <1 : stretching the bin size     
    // start 1 before the first 'in' bin that overlaps the 'out' data.
    // compute by comparing its position with
    // the position of the left edge of the first 'out' bin:
    // left_edge = (out_data.get_min_index() - .5)/zoom + offset
    // x1-.5 <= left_edge < x1+.5
    
    const float inverse_zoom = 1/zoom;
    
    int x2 = out_data.get_min_index();
    int x1 = (int)floor((x2 - .5)*inverse_zoom  + offset + .5);
    
    // the next variable holds the difference between the coordinates
    // of the right edges of the 'in' bin and the 'out' bin, computed
    // in a coordinate system where the 'in' bins are unit distance apart
    double diff_between_right_edges = (x1-offset+.5) - (x2 + .5)*inverse_zoom;
    
    {
      const double diff_between_right1_and_left2 =
	diff_between_right_edges+inverse_zoom;
      if (diff_between_right1_and_left2 > 0 &&
	x1 >= in_data.get_min_index() &&
	x1 <= in_data.get_max_index() )
      {
	out_data[x2] = in_data[x1];
	out_data[x2] *= -diff_between_right1_and_left2;
      }
      else
      {
	if (assign_rest_with_zeroes) 	
	  out_data[x2] *= 0;	  
      }
    }

    for (;
         x1 <= in_data.get_max_index(); 
         x1++, diff_between_right_edges++)
    {
      assert(diff_between_right_edges<= 1/*+epsilon*/);
      assert(diff_between_right_edges>= -inverse_zoom/*-epsilon*/);
      
      if (diff_between_right_edges <= 0)
      {
	// 'in' bin fits entirely in the 'out' bin
	if (x1 >= in_data.get_min_index())
	  out_data[x2] += in_data[x1];
      }
      else 
      {
	// dx = fraction of 'in' bin that lies within 'out' bin
	const double dx= 1- diff_between_right_edges;
	// update current 'out' bin
	if (x1 >= in_data.get_min_index())
	{
	  // out_data[x2] += in_data[x1]*dx;
	  if (fabs(dx) > 1e-5)
	  {
	    out_data[x2] /= dx;
	    out_data[x2] += in_data[x1];
	    out_data[x2] *= dx;
	  }
	}
	// next bin
	x2++;
	diff_between_right_edges -= inverse_zoom;
	// update this one with the rest
	if (x2<= out_data.get_max_index())
	{
	  if (x1 >= in_data.get_min_index())
	  {
	    // out_data[x2] = in_data[x1]*(1-dx);
	    out_data[x2] = in_data[x1];
	    out_data[x2] *= (1-dx);
	  }
	  else if (assign_rest_with_zeroes) 
	  {
	    out_data[x2] *= 0;
	  }
	}
	else
	{
	  // x2 goes out of range, so we can just as well stop
	  break;
	}
      }
      
    }// End of for x1
    
    if (assign_rest_with_zeroes) 
    {
      // set rest of out_data to 0
      for (x2++;
           x2 <= out_data.get_max_index(); 
           x2++)
	out_data[x2] *= 0;
    }
    
    
  }// End of if(zoom>1)
  
}



// Instantiations

template
void 
overlap_interpolate<>(VectorWithOffset<float>& out_data, 
		      const VectorWithOffset<float>& in_data,
		      const float zoom, 
		      const float offset, 
		      const bool assign_rest_with_zeroes);

template
void 
overlap_interpolate<>(VectorWithOffset<double>& out_data, 
		      const VectorWithOffset<double>& in_data,
		      const float zoom, 
		      const float offset, 
		      const bool assign_rest_with_zeroes);

#include "Tensor2D.h"

template
void 
overlap_interpolate<>(VectorWithOffset<Tensor1D<float> >& out_data, 
		      const VectorWithOffset<Tensor1D<float> >& in_data,
		      const float zoom, 
		      const float offset, 
		      const bool assign_rest_with_zeroes);

template
void 
overlap_interpolate<>(VectorWithOffset<Tensor2D<float> >& out_data, 
		      const VectorWithOffset<Tensor2D<float> >& in_data,
		      const float zoom, 
		      const float offset, 
		      const bool assign_rest_with_zeroes);

