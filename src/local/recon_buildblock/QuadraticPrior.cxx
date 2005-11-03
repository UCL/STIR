//
// $Id$
//
/*
    Copyright (C) 2000- $Date$, Hammersmith Imanet Ltd
    This file is part of STIR.

    This file is free software; you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation; either version 2.1 of the License, or
    (at your option) any later version.

    This file is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup recon_buildblock
  \brief  implementation of the stir::QuadraticPrior class 
    
  \author Kris Thielemans
  \author Sanida Mustafovic

  $Date$        
  $Revision$
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
  this->parser.add_key("weights", &weights);
  this->parser.add_key("gradient filename prefix", &gradient_filename_prefix);
  this->parser.add_stop_key("END Quadratic Prior Parameters");
}

template <typename elemT>
bool 
QuadraticPrior<elemT>::post_processing()
{
  if (GeneralisedPrior<elemT>::post_processing()==true)
    return true;
  if (kappa_filename.size() != 0)
    this->kappa_ptr = DiscretisedDensity<3,elemT>::read_from_file(kappa_filename);

  bool warn_about_even_size = false;

  if (this->weights.size() ==0)
    {
      // will call compute_weights() to fill it in
    }
  else
    {
      if (!this->weights.is_regular())
	{
	  warning("Sorry. QuadraticPrior currently only supports regular arrays for the weights");
	  return true;
	}

      const unsigned int size_z = this->weights.size();
      if (size_z%2==0)
	warn_about_even_size = true;
      const int min_index_z = -static_cast<int>(size_z/2);
      this->weights.set_min_index(min_index_z);

      for (int z = min_index_z; z<= this->weights.get_max_index(); ++z)
	{
	  const unsigned int size_y = this->weights[z].size();
	  if (size_y%2==0)
	    warn_about_even_size = true;
	  const int min_index_y = -static_cast<int>(size_y/2);
	  this->weights[z].set_min_index(min_index_y);
	  for (int y = min_index_y; y<= this->weights[z].get_max_index(); ++y)
	    {
	      const unsigned int size_x = this->weights[z][y].size();
	      if (size_x%2==0)
		warn_about_even_size = true;
	      const int min_index_x = -static_cast<int>(size_x/2);
	      this->weights[z][y].set_min_index(min_index_x);
	    }
	}
    }

  if (warn_about_even_size)
    warning("Parsing QuadraticPrior: even number of weights occured in either x,y or z dimension.\n"
	    "I'll (effectively) make this odd by appending a 0 at the end.");
  return false;

}

template <typename elemT>
void
QuadraticPrior<elemT>::set_defaults()
{
  GeneralisedPrior<elemT>::set_defaults();
  this->only_2D = false;
  this->kappa_ptr = 0;  
  this->weights.recycle();
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


  //! get penalty weights for the neigbourhood
template <typename elemT>
Array<3,float>  
QuadraticPrior<elemT>::
get_weights() const
{ return this->weights; }

  //! set penalty weights for the neigbourhood
template <typename elemT>
void 
QuadraticPrior<elemT>::
set_weights(const Array<3,float>& w)
{ this->weights = w; }

  //! get current kappa image
  /*! \warning As this function returns a shared_ptr, this is dangerous. You should not
      modify the image by manipulating the image refered to by this pointer.
      Unpredictable results will occur.
  */
template <typename elemT>
shared_ptr<DiscretisedDensity<3,elemT> >  
QuadraticPrior<elemT>::
get_kappa_sptr() const
{ return this->kappa_ptr; }

  //! set kappa image
template <typename elemT>
void 
QuadraticPrior<elemT>::
set_kappa_sptr(const shared_ptr<DiscretisedDensity<3,elemT> >& k)
{ this->kappa_ptr = k; }


// TODO move to set_up
// initialise to 1/Euclidean distance
static void 
compute_weights(Array<3,float>& weights, const CartesianCoordinate3D<float>& grid_spacing, const bool only_2D)
{
  int min_dz, max_dz;
  if (only_2D)
    {
      min_dz = max_dz = 0;
    }
  else
    {
      min_dz = -1;
      max_dz = 1;
    }
  weights = Array<3,float>(IndexRange3D(min_dz,max_dz,-1,1,-1,1));
  for (int z=min_dz;z<=max_dz;++z)
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

  if (this->weights.get_length() ==0)
  {
    compute_weights(this->weights, current_image_cast.get_grid_spacing(), this->only_2D);
  }
 
 
  
  const bool do_kappa = kappa_ptr.use_count() != 0;
  
  if (do_kappa && kappa_ptr->get_index_range() != current_image_estimate.get_index_range())
    error("QuadraticPrior: kappa image has not the same index range as the reconstructed image\n");

  for (int z=prior_gradient_cast.get_min_z();z<= prior_gradient_cast.get_max_z();z++)
    {
      const int min_dz = max(weights.get_min_index(), prior_gradient_cast.get_min_z()-z);
      const int max_dz = min(weights.get_max_index(), prior_gradient_cast.get_max_z()-z);
	
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
  if (gradient_filename_prefix.size()>0)
    {
      char *filename = new char[gradient_filename_prefix.size()+100];
      sprintf(filename, "%s%d.v", gradient_filename_prefix.c_str(), count);
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
    compute_weights(weights, current_image_cast.get_grid_spacing(), this->only_2D);
  }
 
   
  const bool do_kappa = kappa_ptr.use_count() != 0;
  
  if (do_kappa && kappa_ptr->get_index_range() != current_image_estimate.get_index_range())
    error("QuadraticPrior: kappa image has not the same index range as the reconstructed image\n");

  const int z = coords[1];
  const int y = coords[2];
  const int x = coords[3];
  const int min_dz = max(weights.get_min_index(), prior_Hessian_for_single_densel_cast.get_min_z()-z);
  const int max_dz = min(weights.get_max_index(), prior_Hessian_for_single_densel_cast.get_max_z()-z);
  
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
    compute_weights(weights, current_image_cast.get_grid_spacing(), this->only_2D);
  }  
   
  const bool do_kappa = kappa_ptr.use_count() != 0;
  
  if (do_kappa && kappa_ptr->get_index_range() != current_image_estimate.get_index_range())
    error("QuadraticPrior: kappa image has not the same index range as the reconstructed image\n");

  for (int z=parabolic_surrogate_curvature_cast.get_min_z();z<= parabolic_surrogate_curvature_cast.get_max_z();z++)
    {
      const int min_dz = max(weights.get_min_index(), parabolic_surrogate_curvature_cast.get_min_z()-z);
      const int max_dz = min(weights.get_max_index(), parabolic_surrogate_curvature_cast.get_max_z()-z);
	
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

