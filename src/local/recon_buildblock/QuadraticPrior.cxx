//
// $Id$
//
/*!
  \file
  \ingroup recon_buildblock
  \brief  implementation of the QuadraticPrior class 
    
  \author Kris Thielemans

  $Date$        
  $Revision$
*/
/*
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/

#include "local/stir/recon_buildblock/QuadraticPrior.h"
#include "stir/Succeeded.h"
#include "stir/DiscretisedDensityOnCartesianGrid.h"
#include "stir/IndexRange3D.h"
#include "stir/IndexRange2D.h"

#include "stir/IO/DefaultOutputFileFormat.h"

START_NAMESPACE_STIR

template <typename elemT>
void 
QuadraticPrior<elemT>::initialise_keymap()
{
  GeneralisedPrior<elemT>::initialise_keymap();
  this->parser.add_start_key("Quadratic Prior Parameters");
  this->parser.add_key("only 2D", &only_2D); 
  this->parser.add_key("kappa filename", &kappa_filename);
  this->parser.add_key("precomputed weights", &precomputed_weights);
  this->parser.add_key ("precomputed weights 3D", &precomputed_weights_3D);

  this->parser.add_key("gradient filename", &gradient_filename);
  this->parser.add_stop_key("END Quadratic Prior Parameters");
}


template <typename elemT>
bool 
QuadraticPrior<elemT>::post_processing()
{
  if (GeneralisedPrior<elemT>::post_processing()==true)
    return true;
  if (kappa_filename.size() != 0)
    kappa_ptr = DiscretisedDensity<3,elemT>::read_from_file(kappa_filename);

  if (precomputed_weights.get_length() !=0 || precomputed_weights_3D.get_length() !=0)
  {
     unsigned int size_z=0;
     unsigned int size_y=0;
     unsigned int size_x=0;
    if (precomputed_weights_3D.get_length() !=0)
    {
    size_z = precomputed_weights_3D.get_length();
    size_y = precomputed_weights_3D[0].get_length();
    size_x = precomputed_weights_3D[0][0].get_length();  
    }
    else
    {
      size_z = 1;
     size_y = precomputed_weights.get_length();
     size_x = precomputed_weights[0].get_length();  
    }
    const int min_index_z = -static_cast<int>(size_z/2);
    const int min_index_y = -static_cast<int>(size_y/2);
    const int min_index_x = -static_cast<int>(size_x/2);
    
    weights.grow(IndexRange3D(min_index_z, min_index_z+size_z-1,
			      min_index_y, min_index_y + size_y - 1,
			      min_index_x, min_index_x + size_x - 1 ));

 for (int k = min_index_z; k<= weights.get_max_index(); ++k)
  for (int j = min_index_y; j<= weights[k].get_max_index(); ++j)
    for (int i = min_index_x; i<= weights[k][j].get_max_index(); ++i)
    {
   if (precomputed_weights_3D.get_length() !=0)
   {
    weights[k][j][i] = 
      static_cast<float>(precomputed_weights_3D[k-min_index_z][j-min_index_y][i-min_index_x]);
   }
   else
   {
    weights[0][j][i] = 
      static_cast<float>(precomputed_weights[j-min_index_y][i-min_index_x]);
   }

    }
  }
  return false;

}

template <typename elemT>
void
QuadraticPrior<elemT>::set_defaults()
{
  GeneralisedPrior<elemT>::set_defaults();
  only_2D = false;
  kappa_ptr = 0;  
  precomputed_weights.fill(0);
  precomputed_weights_3D.fill(0);
}

template <>
const char * const 
QuadraticPrior<float>::registered_name =
  "Quadratic";

template <typename elemT>
QuadraticPrior<elemT>::QuadraticPrior()
{
  set_defaults();
}


template <typename elemT>
QuadraticPrior<elemT>::QuadraticPrior(const bool only_2D_v, float penalisation_factor_v)
  :  only_2D(only_2D_v)
{
  this->penalisation_factor = penalisation_factor_v;
}



// TODO move to set_up
// initialise to 1/Euclidean distance
static void 
compute_weights(Array<3,float>& weights, const CartesianCoordinate3D<float>& grid_spacing)
{
  static bool already_computed = false;

  if (already_computed)
    return;

  already_computed = true;
  weights = Array<3,float>(IndexRange3D(-1,1,-1,1,-1,1));
  for (int z=-1;z<=1;++z)
    for (int y=-1;y<=1;++y)
      for (int x=-1;x<=1;++x)
	{
	  if (z==0 && y==0 && x==0)
	    weights[0][0][0] = 0;
	  else
	    {
	      weights[z][y][x] = 
		grid_spacing.x()/
		sqrt(square(x*grid_spacing.x())+
		     square(y*grid_spacing.y())+
		     square(z*grid_spacing.z()));
	    }
	}
}

template <typename elemT>
void 
QuadraticPrior<elemT>::
compute_gradient(DiscretisedDensity<3,elemT>& prior_gradient, 
		 const DiscretisedDensity<3,elemT> &current_image_estimate)
{
  assert(  prior_gradient.get_index_range() == current_image_estimate.get_index_range());  
  if (this->penalisation_factor==0)
  {
    prior_gradient.fill(0);
    return;
  }
  
  
  const DiscretisedDensityOnCartesianGrid<3,float>& current_image_cast =
    dynamic_cast< const DiscretisedDensityOnCartesianGrid<3,float> &>(current_image_estimate);
  
  DiscretisedDensityOnCartesianGrid<3,float>& prior_gradient_cast =
    dynamic_cast<DiscretisedDensityOnCartesianGrid<3,float> &>(prior_gradient);

  if (weights.get_length() ==0)
  {
    compute_weights(weights, current_image_cast.get_grid_spacing());
  }
 
 
  
  const bool do_kappa = kappa_ptr.use_count() != 0;
  
  if (do_kappa && kappa_ptr->get_index_range() != current_image_estimate.get_index_range())
    error("QuadraticPrior: kappa image has not the same index range as the reconstructed image\n");

  for (int z=prior_gradient_cast.get_min_z();z<= prior_gradient_cast.get_max_z();z++)
    {
	int min_dz, max_dz;
	if (only_2D)
	  {
	    min_dz = max_dz = 0;
	  }
	else
	  {
	    min_dz = max(weights.get_min_index(), prior_gradient_cast.get_min_z()-z);
	    max_dz = min(weights.get_max_index(), prior_gradient_cast.get_max_z()-z);
	  }
	
	for (int y=prior_gradient_cast.get_min_y();y<= prior_gradient_cast.get_max_y();y++)
	  {
	    const int min_dy = max(weights[0].get_min_index(), prior_gradient_cast.get_min_y()-y);
	    const int max_dy = min(weights[0].get_max_index(), prior_gradient_cast.get_max_y()-y);	    

	    for (int x=prior_gradient_cast.get_min_x();x<= prior_gradient_cast.get_max_x();x++)       
	      {
		const int min_dx = max(weights[0][0].get_min_index(), prior_gradient_cast.get_min_x()-x);
		const int max_dx = min(weights[0][0].get_max_index(), prior_gradient_cast.get_max_x()-x);
		
		/* formula:
		  sum_dx,dy,dz
		   weights[dz][dy][dx] *
		   (current_image_estimate[z][y][x] - current_image_estimate[z+dz][y+dy][x+dx]) *
		   (*kappa_ptr)[z][y][x] * (*kappa_ptr)[z+dz][y+dy][x+dx];
		*/
#if 1
		elemT gradient = 0;
		for (int dz=min_dz;dz<=max_dz;++dz)
		  for (int dy=min_dy;dy<=max_dy;++dy)
		    for (int dx=min_dx;dx<=max_dx;++dx)
		      {
			elemT current =
			  weights[dz][dy][dx] *
			  (current_image_estimate[z][y][x] - current_image_estimate[z+dz][y+dy][x+dx]);

			if (do_kappa)
			  current *= 
			    (*kappa_ptr)[z][y][x] * (*kappa_ptr)[z+dz][y+dy][x+dx];

			gradient += current;
		      }
#else
		// attempt to speed up by precomputing the sum of weights.
		// The current code gives identical results but is actually slower
		// than the above, at least when kappas are present.


		// precompute sum of weights
		// TODO without kappas, this is just weights.sum() most of the time, 
		// but not near edges
		float sum_of_weights = 0;
		{
		  if (do_kappa)
		    {		     
		      for (int dz=min_dz;dz<=max_dz;++dz)
			for (int dy=min_dy;dy<=max_dy;++dy)
			  for (int dx=min_dx;dx<=max_dx;++dx)
			    sum_of_weights +=  weights[dz][dy][dx]*(*kappa_ptr)[z+dz][y+dy][x+dx];
		    }
		  else
		    {
		      for (int dz=min_dz;dz<=max_dz;++dz)
			for (int dy=min_dy;dy<=max_dy;++dy)
			  for (int dx=min_dx;dx<=max_dx;++dx)
			    sum_of_weights +=  weights[dz][dy][dx];
		    }
		}
		// now compute contribution of central term
		elemT gradient = sum_of_weights * current_image_estimate[z][y][x] ;

		// subtract the rest
		for (int dz=min_dz;dz<=max_dz;++dz)
		  for (int dy=min_dy;dy<=max_dy;++dy)
		    for (int dx=min_dx;dx<=max_dx;++dx)
		      {
			elemT current =
			  weights[dz][dy][dx] * current_image_estimate[z+dz][y+dy][x+dx];

			if (do_kappa)
			  current *= (*kappa_ptr)[z+dz][y+dy][x+dx];

			gradient -= current;
		      }
		// multiply with central kappa
		if (do_kappa)
		  gradient *= (*kappa_ptr)[z][y][x];
#endif
		prior_gradient[z][y][x]= gradient * this->penalisation_factor;
	      }              
	  }
    }

  std::cerr << "Prior gradient max " << prior_gradient.find_max()
    << ", min " << prior_gradient.find_min() << std::endl;

  static int count = 0;
  ++count;
  if (gradient_filename.size()>0)
    {
      char *filename = new char[gradient_filename.size()+100];
      sprintf(filename, "%s%d.v", gradient_filename.c_str(), count);
      DefaultOutputFileFormat output_file_format;
      output_file_format.
	write_to_file(filename, prior_gradient);
      delete filename;
    }
}

template <typename elemT>
void 
QuadraticPrior<elemT>::
compute_Hessian(DiscretisedDensity<3,elemT>& prior_Hessian_for_single_densel, 
		const BasicCoordinate<3,int>& coords,
		const DiscretisedDensity<3,elemT> &current_image_estimate)
{
  assert(  prior_Hessian_for_single_densel.get_index_range() == current_image_estimate.get_index_range());  
  prior_Hessian_for_single_densel.fill(0);
  if (this->penalisation_factor==0)
  {
    return;
  }
  
  
  const DiscretisedDensityOnCartesianGrid<3,float>& current_image_cast =
    dynamic_cast< const DiscretisedDensityOnCartesianGrid<3,float> &>(current_image_estimate);
  
  DiscretisedDensityOnCartesianGrid<3,float>& prior_Hessian_for_single_densel_cast =
    dynamic_cast<DiscretisedDensityOnCartesianGrid<3,float> &>(prior_Hessian_for_single_densel);

  if (weights.get_length() ==0)
  {
    compute_weights(weights, current_image_cast.get_grid_spacing());
  }
 
   
  const bool do_kappa = kappa_ptr.use_count() != 0;
  
  if (do_kappa && kappa_ptr->get_index_range() != current_image_estimate.get_index_range())
    error("QuadraticPrior: kappa image has not the same index range as the reconstructed image\n");

  const int z = coords[1];
  const int y = coords[2];
  const int x = coords[3];
  int min_dz, max_dz;
  if (only_2D)
  {
    min_dz = max_dz = 0;
  }
  else
  {
    min_dz = max(weights.get_min_index(), prior_Hessian_for_single_densel_cast.get_min_z()-z);
    max_dz = min(weights.get_max_index(), prior_Hessian_for_single_densel_cast.get_max_z()-z);
  }
  // TODO use z,y,x
  const int min_dy = max(weights[0].get_min_index(), prior_Hessian_for_single_densel_cast.get_min_y()-y);
  const int max_dy = min(weights[0].get_max_index(), prior_Hessian_for_single_densel_cast.get_max_y()-y);
  
  const int min_dx = max(weights[0][0].get_min_index(), prior_Hessian_for_single_densel_cast.get_min_x()-x);
  const int max_dx = min(weights[0][0].get_max_index(), prior_Hessian_for_single_densel_cast.get_max_x()-x);
  
  elemT diagonal = 0;
  for (int dz=min_dz;dz<=max_dz;++dz)
    for (int dy=min_dy;dy<=max_dy;++dy)
      for (int dx=min_dx;dx<=max_dx;++dx)
      {
	// dz==0,dy==0,dx==0 will have weight 0, so we can just include it in the loop
	elemT current =
	  weights[dz][dy][dx];
	
	if (do_kappa)
	  current *= 
	  (*kappa_ptr)[z][y][x] * (*kappa_ptr)[z+dz][y+dy][x+dx];
	
	diagonal += current;
	prior_Hessian_for_single_densel_cast[z+dz][y+dy][x+dx] = -current*this->penalisation_factor;
      }
      
      prior_Hessian_for_single_densel[z][y][x]= diagonal * this->penalisation_factor;
}              

template <typename elemT>
void 
QuadraticPrior<elemT>::parabolic_surrogate_curvature(DiscretisedDensity<3,elemT>& parabolic_surrogate_curvature, 
			const DiscretisedDensity<3,elemT> &current_image_estimate)
{

  assert( parabolic_surrogate_curvature.get_index_range() == current_image_estimate.get_index_range());  
  if (this->penalisation_factor==0)
  {
    parabolic_surrogate_curvature.fill(0);
    return;
  }
  
  
  const DiscretisedDensityOnCartesianGrid<3,float>& current_image_cast =
    dynamic_cast< const DiscretisedDensityOnCartesianGrid<3,float> &>(current_image_estimate);
  
  DiscretisedDensityOnCartesianGrid<3,float>& parabolic_surrogate_curvature_cast =
    dynamic_cast<DiscretisedDensityOnCartesianGrid<3,float> &>(parabolic_surrogate_curvature);

  if (weights.get_length() ==0)
  {
    compute_weights(weights, current_image_cast.get_grid_spacing());
  }  
   
  const bool do_kappa = kappa_ptr.use_count() != 0;
  
  if (do_kappa && kappa_ptr->get_index_range() != current_image_estimate.get_index_range())
    error("QuadraticPrior: kappa image has not the same index range as the reconstructed image\n");

  for (int z=parabolic_surrogate_curvature_cast.get_min_z();z<= parabolic_surrogate_curvature_cast.get_max_z();z++)
    {
	int min_dz, max_dz;
	if (only_2D)
	  {
	    min_dz = max_dz = 0;
	  }
	else
	  {
	    min_dz = max(weights.get_min_index(), parabolic_surrogate_curvature_cast.get_min_z()-z);
	    max_dz = min(weights.get_max_index(), parabolic_surrogate_curvature_cast.get_max_z()-z);
	  }
	
	for (int y=parabolic_surrogate_curvature_cast.get_min_y();y<= parabolic_surrogate_curvature_cast.get_max_y();y++)
	  {
	    const int min_dy = max(weights[0].get_min_index(), parabolic_surrogate_curvature_cast.get_min_y()-y);
	    const int max_dy = min(weights[0].get_max_index(), parabolic_surrogate_curvature_cast.get_max_y()-y);

	    for (int x=parabolic_surrogate_curvature_cast.get_min_x();x<= parabolic_surrogate_curvature_cast.get_max_x();x++)       
	      {  	        	     
		const int min_dx = max(weights[0][0].get_min_index(), parabolic_surrogate_curvature_cast.get_min_x()-x);
		const int max_dx = min(weights[0][0].get_max_index(), parabolic_surrogate_curvature_cast.get_max_x()-x);
		
		elemT gradient = 0;
		for (int dz=min_dz;dz<=max_dz;++dz)
		  for (int dy=min_dy;dy<=max_dy;++dy)
		    for (int dx=min_dx;dx<=max_dx;++dx)
		      {
			// 1 comes from omega = psi'(t)/t = 2*t/2t =1		       
			elemT current =
			  weights[dz][dy][dx] *1;

			 if (do_kappa)
			  current *= 
			    (*kappa_ptr)[z][y][x] * (*kappa_ptr)[z+dz][y+dy][x+dx];

			gradient += current;
		      }
		
		parabolic_surrogate_curvature[z][y][x]= gradient * this->penalisation_factor;
	      }              
	  }
    }

  std::cerr << " parabolic_surrogate_curvature max " << parabolic_surrogate_curvature.find_max()
    << ", min " << parabolic_surrogate_curvature.find_min() << std::endl;
  /*{
    static int count = 0;
    ++count;
    char filename[20];
    sprintf(filename, "normalised_gradient%d.v",count);
    write_basic_interfile(filename, parabolic_surrogate_curvature);
  }*/
}

#  ifdef _MSC_VER
// prevent warning message on reinstantiation, 
// note that we get a linking error if we don't have the explicit instantiation below
#  pragma warning(disable:4660)
#  endif


template class QuadraticPrior<float>;

END_NAMESPACE_STIR

