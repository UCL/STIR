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
#include "stir/interfile.h"

START_NAMESPACE_STIR

template <typename elemT>
void 
QuadraticPrior<elemT>::initialise_keymap()
{
  GeneralisedPrior<elemT>::initialise_keymap();
  parser.add_start_key("Quadratic Prior Parameters");
  parser.add_key("only 2D", &only_2D); 
  parser.add_key("kappa filename", &kappa_filename);
  parser.add_stop_key("END Quadratic Prior Parameters");
}


template <typename elemT>
bool 
QuadraticPrior<elemT>::post_processing()
{
  if (GeneralisedPrior<elemT>::post_processing()==true)
    return true;
  if (kappa_filename.size() != 0)
    kappa_ptr = DiscretisedDensity<3,elemT>::read_from_file(kappa_filename);
  return false;
}

template <typename elemT>
void
QuadraticPrior<elemT>::set_defaults()
{
  GeneralisedPrior<elemT>::set_defaults();
  only_2D = false;
  kappa_ptr = 0;  
}


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
  penalisation_factor = penalisation_factor_v;
}

// TODO move to set_up
// initialise to 1/Euclidean distance
static void 
compute_weights(Array<3,float>& weights, const CartesianCoordinate3D<float>& grid_spacing)
{
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
  if (penalisation_factor==0)
  {
    prior_gradient.fill(0);
    return;
  }
  
  
  const DiscretisedDensityOnCartesianGrid<3,float>& current_image_cast =
    dynamic_cast< const DiscretisedDensityOnCartesianGrid<3,float> &>(current_image_estimate);
  
  DiscretisedDensityOnCartesianGrid<3,float>& prior_gradient_cast =
    dynamic_cast<DiscretisedDensityOnCartesianGrid<3,float> &>(prior_gradient);

  compute_weights(weights, current_image_cast.get_grid_spacing());
 
  
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
	    min_dz = max(-1, prior_gradient_cast.get_min_z()-z);
	    max_dz = min(1, prior_gradient_cast.get_max_z()-z);
	  }
	
	for (int y=prior_gradient_cast.get_min_y();y<= prior_gradient_cast.get_max_y();y++)
	  {
	    int min_dy, max_dy;
	    {
	      min_dy = max(-1, prior_gradient_cast.get_min_y()-y);
	      max_dy = min(1, prior_gradient_cast.get_max_y()-y);
	    }

	    for (int x=prior_gradient_cast.get_min_x();x<= prior_gradient_cast.get_max_x();x++)       
	      {
		int min_dx, max_dx;
		{
		  min_dx = max(-1, prior_gradient_cast.get_min_x()-x);
		  max_dx = min(1, prior_gradient_cast.get_max_x()-x);
		}
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
		
		prior_gradient[z][y][x]= gradient * penalisation_factor;
	      }              
	  }
    }

  std::cerr << "Prior gradient max " << prior_gradient.find_max()
    << ", min " << prior_gradient.find_min() << std::endl;
 /* {
    static int count = 0;
    ++count;
    char filename[20];
    sprintf(filename, "gradient%d.v",count);
    write_basic_interfile(filename, prior_gradient);
  }*/
}

template <typename elemT>
void 
QuadraticPrior<elemT>::parabolic_surrogate_curvature(DiscretisedDensity<3,elemT>& parabolic_surrogate_curvature, 
			const DiscretisedDensity<3,elemT> &current_image_estimate)
{

  assert( parabolic_surrogate_curvature.get_index_range() == current_image_estimate.get_index_range());  
  if (penalisation_factor==0)
  {
    parabolic_surrogate_curvature.fill(0);
    return;
  }
  
  
  const DiscretisedDensityOnCartesianGrid<3,float>& current_image_cast =
    dynamic_cast< const DiscretisedDensityOnCartesianGrid<3,float> &>(current_image_estimate);
  
  DiscretisedDensityOnCartesianGrid<3,float>& parabolic_surrogate_curvature_cast =
    dynamic_cast<DiscretisedDensityOnCartesianGrid<3,float> &>(parabolic_surrogate_curvature);

  compute_weights(weights, current_image_cast.get_grid_spacing());
  
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
	    min_dz = max(-1, parabolic_surrogate_curvature_cast.get_min_z()-z);
	    max_dz = min(1, parabolic_surrogate_curvature_cast.get_max_z()-z);
	  }
	
	for (int y=parabolic_surrogate_curvature_cast.get_min_y();y<= parabolic_surrogate_curvature_cast.get_max_y();y++)
	  {
	    int min_dy, max_dy;
	    {
	      min_dy = max(-1, parabolic_surrogate_curvature_cast.get_min_y()-y);
	      max_dy = min(1, parabolic_surrogate_curvature_cast.get_max_y()-y);
	    }

	    for (int x=parabolic_surrogate_curvature_cast.get_min_x();x<= parabolic_surrogate_curvature_cast.get_max_x();x++)       
	      {
		int min_dx, max_dx;
		{
		  min_dx = max(-1, parabolic_surrogate_curvature_cast.get_min_x()-x);
		  max_dx = min(1, parabolic_surrogate_curvature_cast.get_max_x()-x);
		}
		elemT gradient = 0;
		for (int dz=min_dz;dz<=max_dz;++dz)
		  for (int dy=min_dy;dy<=max_dy;++dy)
		    for (int dx=min_dx;dx<=max_dx;++dx)
		      {
			// 1 comes form omega = psi'(t)/t = 2*t/2t =1		       
			elemT current =
			  weights[dz][dy][dx] *1;

			 if (do_kappa)
			  current *= 
			    (*kappa_ptr)[z][y][x] * (*kappa_ptr)[z+dz][y+dy][x+dx];

			gradient += current;
		      }
		
		parabolic_surrogate_curvature[z][y][x]= gradient * penalisation_factor;
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


template QuadraticPrior<float>;

END_NAMESPACE_STIR

