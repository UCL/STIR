/*
    Copyright (C) 2000- 2011, Hammersmith Imanet Ltd
	Copyright (C) 2017, University College London
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
  \ingroup priors
  \brief  implementation of the stir::QuadraticBowsherPrior class 
    
  \author Kris Thielemans
  \author Sanida Mustafovic
*/

#include "stir/recon_buildblock/QuadraticBowsherPrior.h"
#include "stir/Succeeded.h"
#include "stir/DiscretisedDensityOnCartesianGrid.h"
#include "stir/IndexRange3D.h"
#include "stir/IO/OutputFileFormat.h"
#include "stir/IO/read_from_file.h"
#include "stir/is_null_ptr.h"
#include "stir/recon_buildblock/IterativeReconstruction.h"
#include "stir/shared_ptr.h"
#include "stir/recon_buildblock/IterativeReconstruction.h"
#include <algorithm>

/* Pretty horrible code because we don't have an iterator of neigbhourhoods yet
 */

START_NAMESPACE_STIR

template <typename elemT>
void 
QuadraticBowsherPrior<elemT>::initialise_keymap()
{
  base_type::initialise_keymap();
  this->parser.add_start_key("NewAnatomical Prior Parameters");
  this->parser.add_key("only 2D", &only_2D); 
  this->parser.add_key("kappa filename", &kappa_filename);
  this->parser.add_key("anatomical image filename",&anatomical_image_filename);
  this->parser.add_key("nmax", &nmax);
#if 0
  this->parser.add_key("weights", &weights);
#endif
  this->parser.add_key("gradient filename prefix", &gradient_filename_prefix);
  this->parser.add_stop_key("END NewAnatomical Prior Parameters");
}

template <typename elemT>
bool 
QuadraticBowsherPrior<elemT>::post_processing()
{
  if (base_type::post_processing()==true)
    return true;
  if (kappa_filename.size() != 0)
    this->kappa_ptr = read_from_file<DiscretisedDensity<3,elemT> >(kappa_filename);
  this->anatomical_image_ptr = read_from_file<DiscretisedDensity<3,elemT> >(anatomical_image_filename);
 
  bool warn_about_even_size = false;

  if (this->weights.size() ==0)
    {
      // will call compute_weights() to fill it in
    }
  else
    {
      error("TODO");
#if 0
      if (!this->weights.is_regular())
        {
          warning("Sorry. QuadraticBowsherPrior currently only supports regular arrays for the weights");
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
#endif
    }

  if (warn_about_even_size)
    warning("Parsing QuadraticBowsherPrior: even number of weights occured in either x,y or z dimension.\n"
            "I'll (effectively) make this odd by appending a 0 at the end.");
  return false;

}

template <typename elemT>
void
QuadraticBowsherPrior<elemT>::set_defaults()
{
  base_type::set_defaults();
  this->only_2D = false;
  this->kappa_ptr.reset();  
  this->weights.recycle();
  //this->anatomical_image_ptr.reset();
}

template <>
const char * const 
QuadraticBowsherPrior<float>::registered_name =
  "NewAnatomical";

template <typename elemT>
QuadraticBowsherPrior<elemT>::QuadraticBowsherPrior()
{
  set_defaults();
}


template <typename elemT>
QuadraticBowsherPrior<elemT>::QuadraticBowsherPrior(const bool only_2D_v, float penalisation_factor_v)
  :  only_2D(only_2D_v)
{
  this->penalisation_factor = penalisation_factor_v;
}


  //! get penalty weights for the neigbourhood
template <typename elemT>
Array<6,float>  
QuadraticBowsherPrior<elemT>::
get_weights() const
{ return this->weights; }

  //! set penalty weights for the neigbourhood
template <typename elemT>
void 
QuadraticBowsherPrior<elemT>::
set_weights(const Array<6,float>& w)
{ this->weights = w; }

  //! get current kappa image
  /*! \warning As this function returns a shared_ptr, this is dangerous. You should not
      modify the image by manipulating the image refered to by this pointer.
      Unpredictable results will occur.
  */
template <typename elemT>
shared_ptr<DiscretisedDensity<3,elemT> >  
QuadraticBowsherPrior<elemT>::
get_kappa_sptr() const
{ return this->kappa_ptr; }

//template <typename elemT>
//Array<6,float>
//QuadraticBowsherPrior<elemT>::
//get_anatomical_image() const
//{ return this-> anatomical_image_ptr; }

  //! set kappa image
template <typename elemT>
void 
QuadraticBowsherPrior<elemT>::
set_kappa_sptr(const shared_ptr<DiscretisedDensity<3,elemT> >& k)
{ this->kappa_ptr = k; }

//template <typename elemT>
//void 
//QuadraticBowsherPrior<elemT>::
//set_anatomical_image(const Array<3,float>& an)
//{ this->anatomical_image_ptr = an; }
/*
  //! initialise anatomical image
template <typename elemT>
shared_ptr<DiscretisedDensity<3,elemT> > 
QuadraticBowsherPrior<elemT>::
get_anatomical_image_sptr() const
{ return this->anatomical_image_ptr; }

  //! set anatomical image
template <typename elemT>
void 
QuadraticBowsherPrior<elemT>::
set_anatomical_image_sptr(const shared_ptr<DiscretisedDensity<3,elemT> >& an)
{ this->anatomical_image_ptr = an; }
*/

// TODO move to set_up
// initialise to 1/Euclidean distance
static void 
compute_weights(Array<6,float>& weights, const IndexRange<3>& image_index_range, const CartesianCoordinate3D<float>& grid_spacing, const bool only_2D, const shared_ptr<DiscretisedDensity<3,float> >& anatomical_image_ptr, const int nmax)
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

  BasicCoordinate<3, int> min_image, max_image;
  image_index_range.get_regular_range(min_image, max_image);
  weights = Array<6,float>(IndexRange<6>(make_coordinate(min_image[1],min_image[2],min_image[3],min_dz,-1,-1),
					 make_coordinate(max_image[1],max_image[2],max_image[3],max_dz,1,1)));
  /*
  IndexRange<3> anatomic_range = anatomical_image_ptr->get_index_range();
  CartesianCoordinate3D<float> anatomic_grid = anatomical_image_ptr->get_origin(); 

  //BasicCoordinate<3, int> tmp;
  //tmp = anatomical_image_ptr->get_index_coordinates_for_physical_coordinates(anatomic_grid);

  Array<3,float> prova = Array<3,float>(anatomic_range);
  */
  //Array<3,float> prova = anatomical_image_ptr->at(anatomic_grid);
  //Array<3,int> anatomical_image;
  //anatomical_image = Array<3,float>(anatomic_range);
  //BasicCoordinate<3,int> min_anatomic, max_anatomic;
  //anatomic_range.get_regular_range(min_anatomic,max_anatomic);
  //DiscretisedDensityOnCartesianGrid<3,float> new_grid_spacing = anatomical_image_ptr->clone();
  //  set_grid_spacing(new_grid_spacing);
  
  
  const int min_z = min_image[1];
  const int max_z = max_image[1];
  // vector to store anatomical difference in the neighborhood for one voxel
  std::vector<float> anatomical_diff((max_dz-min_dz+1)*3*3);
  /*
  for (int z=min_dz;z<=max_dz; ++z)
    for (int y=-1;y<=1; ++y)
      for (int x=-1;x<=1; ++x)
	{
	    anatomical_weights[z] = prova.z() ;
	    anatomical_weights[z][y] = prova.y;
	    anatomical_weights[z][y][x] = prova.x();
	    }*/
 

  for (int z=min_z; z<=max_z; z++)
    {        
      const int min_y = weights[z].get_min_index();
      const int max_y = weights[z].get_max_index();

      for (int y=min_y;y<= max_y;y++)
	{
	  const int min_x = weights[z][y].get_min_index(); 
	  const int max_x = weights[z][y].get_max_index(); 

	  for (int x=min_x;x<= max_x;x++)       
	    {
	      /* min_index <= z+ dz <= max_index
	       */
	      const int min_dz = max(weights[z][y][x].get_min_index(), min_z-z);
	      const int max_dz = min(weights[z][y][x].get_max_index(), max_z-z);
	      const int min_dy = max(weights[z][y][x][0].get_min_index(), min_y-y);
	      const int max_dy = min(weights[z][y][x][0].get_max_index(), max_y-y);            
	      const int min_dx = max(weights[z][y][x][0][0].get_min_index(), min_x-x);
	      const int max_dx = min(weights[z][y][x][0][0].get_max_index(), max_x-x);
		

	      int anatomical_idx = 0;
	      for (int dz=min_dz;dz<=max_dz;++dz)
		for (int dy=min_dy;dy<=max_dy;++dy)
		  for (int dx=min_dx;dx<=max_dx;++dx)
		    {
		      anatomical_diff[anatomical_idx++]=abs((*anatomical_image_ptr)[z][y][x] - (*anatomical_image_ptr)[z+dz][y+dy][x+dx]);
		      if (dz==0 && dy==0 && dx==0)
			{
			  weights[z][y][x][0][0][0] = 0;
			}	    
		      else
			{
			  weights[z][y][x][dz][dy][dx] = 
			    grid_spacing.x()/
			    sqrt(square(dx*grid_spacing.x())+
				 square(dy*grid_spacing.y())+
				 square(dz*grid_spacing.z()));
		    
			}
		    }
	      // find nmax-th element in the range that we actually filled in
	      std::nth_element(anatomical_diff.begin(),
			       anatomical_diff.begin() + nmax,
			       anatomical_diff.begin() + anatomical_idx);
	      const float threshold = *(anatomical_diff.begin() + nmax);
	      anatomical_idx = 0;
	      for (int dz=min_dz;dz<=max_dz;++dz)
		for (int dy=min_dy;dy<=max_dy;++dy)
		  for (int dx=min_dx;dx<=max_dx;++dx)
		    {
		      if (anatomical_diff[anatomical_idx++]>threshold)
			weights[z][y][x][dz][dy][dx] = 0.F;
		    }
	    }
	}
    }
}


template <typename elemT>
double
QuadraticBowsherPrior<elemT>::
compute_value(const DiscretisedDensity<3,elemT> &current_image_estimate)
{
  if (this->penalisation_factor==0)
  {
    return 0.;
  }
  
  
  const DiscretisedDensityOnCartesianGrid<3,elemT>& current_image_cast =
    dynamic_cast< const DiscretisedDensityOnCartesianGrid<3,elemT> &>(current_image_estimate);
  
  if (this->weights.get_length() ==0)
  {
    compute_weights(this->weights,  current_image_cast.get_index_range(), current_image_cast.get_grid_spacing(), this->only_2D, this->anatomical_image_ptr, this->nmax);
  }
    
  const bool do_kappa = !is_null_ptr(kappa_ptr);
  
  if (do_kappa && !kappa_ptr->has_same_characteristics(current_image_estimate))
    error("QuadraticBowsherPrior: kappa image has not the same index range as the reconstructed image\n");


  double result = 0.;
  const int min_z = current_image_estimate.get_min_index(); 
  const int max_z = current_image_estimate.get_max_index(); 
  for (int z=min_z; z<=max_z; z++)
    {
      
      const int min_y = current_image_estimate[z].get_min_index();
      const int max_y = current_image_estimate[z].get_max_index();

        for (int y=min_y;y<= max_y;y++)
          {

            const int min_x = current_image_estimate[z][y].get_min_index(); 
            const int max_x = current_image_estimate[z][y].get_max_index(); 

            for (int x=min_x;x<= max_x;x++)       
              {
		const int min_dz = max(weights[z][y][x].get_min_index(), min_z-z);
		const int max_dz = min(weights[z][y][x].get_max_index(), max_z-z);
		const int min_dy = max(weights[z][y][x][0].get_min_index(), min_y-y);
		const int max_dy = min(weights[z][y][x][0].get_max_index(), max_y-y);            
		const int min_dx = max(weights[z][y][x][0][0].get_min_index(), min_x-x);
                const int max_dx = min(weights[z][y][x][0][0].get_max_index(), max_x-x);
                
                /* formula:
                  sum_dx,dy,dz
                   1/2 weights[dz][dy][dx] *
                   (current_image_estimate[z][y][x] - current_image_estimate[z+dz][y+dy][x+dx])^2 *
                   (*kappa_ptr)[z][y][x] * (*kappa_ptr)[z+dz][y+dy][x+dx];
                */
                for (int dz=min_dz;dz<=max_dz;++dz)
                  for (int dy=min_dy;dy<=max_dy;++dy)
                    for (int dx=min_dx;dx<=max_dx;++dx)
                      {
                        elemT current =
                          weights[z][y][x][dz][dy][dx] *
                          square(current_image_estimate[z][y][x] - current_image_estimate[z+dz][y+dy][x+dx])/2;

                        if (do_kappa)
                          current *= 
                            (*kappa_ptr)[z][y][x] * (*kappa_ptr)[z+dz][y+dy][x+dx];

                        result += static_cast<double>(current);
                      }
              }              
          }
    }
  return result * this->penalisation_factor;
}

template <typename elemT>
void 
QuadraticBowsherPrior<elemT>::
compute_gradient(DiscretisedDensity<3,elemT>& prior_gradient, 
                 const DiscretisedDensity<3,elemT> &current_image_estimate)
{
  assert(  prior_gradient.has_same_characteristics(current_image_estimate));  
  if (this->penalisation_factor==0)
  {
    prior_gradient.fill(0);
    return;
  }
  
  
  const DiscretisedDensityOnCartesianGrid<3,elemT>& current_image_cast =
    dynamic_cast< const DiscretisedDensityOnCartesianGrid<3,elemT> &>(current_image_estimate);
  
  if (this->weights.get_length() ==0)
  {
    compute_weights(this->weights, current_image_cast.get_index_range(), current_image_cast.get_grid_spacing(), this->only_2D, this->anatomical_image_ptr, this->nmax);
  }
 
 
  
  const bool do_kappa = !is_null_ptr(kappa_ptr);
  if (do_kappa && !kappa_ptr->has_same_characteristics(current_image_estimate))
    error("QuadraticBowsherPrior: kappa image has not the same index range as the reconstructed image\n");

  const int min_z = current_image_estimate.get_min_index();  
  const int max_z = current_image_estimate.get_max_index();  
  for (int z=min_z; z<=max_z; z++) 
    { 
      
      const int min_y = current_image_estimate[z].get_min_index(); 
      const int max_y = current_image_estimate[z].get_max_index(); 
      
      for (int y=min_y;y<= max_y;y++) 
        { 

          const int min_x = current_image_estimate[z][y].get_min_index();
          const int max_x = current_image_estimate[z][y].get_max_index();  
          
          for (int x=min_x;x<= max_x;x++)
            {
	      const int min_dz = max(weights[z][y][x].get_min_index(), min_z-z); 
	      const int max_dz = min(weights[z][y][x].get_max_index(), max_z-z); 
	      const int min_dy = max(weights[z][y][x][0].get_min_index(), min_y-y); 
	      const int max_dy = min(weights[z][y][x][0].get_max_index(), max_y-y); 
              const int min_dx = max(weights[z][y][x][0][0].get_min_index(), min_x-x);
              const int max_dx = min(weights[z][y][x][0][0].get_max_index(), max_x-x);
                
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
                          weights[z][y][x][dz][dy][dx] *
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
                            sum_of_weights +=  weights[z][y][x][dz][dy][dx]*(*kappa_ptr)[z+dz][y+dy][x+dx];
                    }
                  else
                    {
                      for (int dz=min_dz;dz<=max_dz;++dz)
                        for (int dy=min_dy;dy<=max_dy;++dy)
                          for (int dx=min_dx;dx<=max_dx;++dx)
                            sum_of_weights +=  weights[z][y][x][dz][dy][dx];
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
                          weights[z][y][x][dz][dy][dx] * current_image_estimate[z+dz][y+dy][x+dx];

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
      OutputFileFormat<DiscretisedDensity<3,elemT> >::default_sptr()->
        write_to_file(filename, prior_gradient);
      delete[] filename;
    }
}

template <typename elemT>
void 
QuadraticBowsherPrior<elemT>::
compute_Hessian(DiscretisedDensity<3,elemT>& prior_Hessian_for_single_densel, 
                const BasicCoordinate<3,int>& coords,
                const DiscretisedDensity<3,elemT> &current_image_estimate)
{
  assert(  prior_Hessian_for_single_densel.has_same_characteristics(current_image_estimate));
  prior_Hessian_for_single_densel.fill(0);
  if (this->penalisation_factor==0)
  {
    return;
  }
  
  
  const DiscretisedDensityOnCartesianGrid<3,elemT>& current_image_cast =
    dynamic_cast< const DiscretisedDensityOnCartesianGrid<3,elemT> &>(current_image_estimate);
  
  DiscretisedDensityOnCartesianGrid<3,elemT>& prior_Hessian_for_single_densel_cast =
    dynamic_cast<DiscretisedDensityOnCartesianGrid<3,elemT> &>(prior_Hessian_for_single_densel);

  if (weights.get_length() ==0)
  {
    compute_weights(weights, current_image_cast.get_index_range(), current_image_cast.get_grid_spacing(), this->only_2D, this->anatomical_image_ptr, this->nmax);
  }
 
   
  const bool do_kappa = !is_null_ptr(kappa_ptr);
  
  if (do_kappa && kappa_ptr->has_same_characteristics(current_image_estimate))
    error("QuadraticBowsherPrior: kappa image has not the same index range as the reconstructed image\n");

  const int z = coords[1];
  const int y = coords[2];
  const int x = coords[3];
  const int min_dz = max(weights[z][y][x].get_min_index(), prior_Hessian_for_single_densel.get_min_index()-z);
  const int max_dz = min(weights[z][y][x].get_max_index(), prior_Hessian_for_single_densel.get_max_index()-z);
  
  const int min_dy = max(weights[z][y][x][0].get_min_index(), prior_Hessian_for_single_densel[z].get_min_index()-y);
  const int max_dy = min(weights[z][y][x][0].get_max_index(), prior_Hessian_for_single_densel[z].get_max_index()-y);
  
  const int min_dx = max(weights[z][y][x][0][0].get_min_index(), prior_Hessian_for_single_densel[z][y].get_min_index()-x);
  const int max_dx = min(weights[z][y][x][0][0].get_max_index(), prior_Hessian_for_single_densel[z][y].get_max_index()-x);
  
  elemT diagonal = 0;
  for (int dz=min_dz;dz<=max_dz;++dz)
    for (int dy=min_dy;dy<=max_dy;++dy)
      for (int dx=min_dx;dx<=max_dx;++dx)
      {
        // dz==0,dy==0,dx==0 will have weight 0, so we can just include it in the loop
        elemT current =
          weights[z][y][x][dz][dy][dx];
        
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
QuadraticBowsherPrior<elemT>::parabolic_surrogate_curvature(DiscretisedDensity<3,elemT>& parabolic_surrogate_curvature, 
                        const DiscretisedDensity<3,elemT> &current_image_estimate)
{

  assert( parabolic_surrogate_curvature.has_same_characteristics(current_image_estimate));
  if (this->penalisation_factor==0)
  {
    parabolic_surrogate_curvature.fill(0);
    return;
  }
  
  
  const DiscretisedDensityOnCartesianGrid<3,elemT>& current_image_cast =
    dynamic_cast< const DiscretisedDensityOnCartesianGrid<3,elemT> &>(current_image_estimate);
  
  if (weights.get_length() ==0)
  {
    compute_weights(weights, current_image_cast.get_index_range(), current_image_cast.get_grid_spacing(), this->only_2D, this->anatomical_image_ptr, this->nmax);
  }  
   
  const bool do_kappa = !is_null_ptr(kappa_ptr);
  
  if (do_kappa && !kappa_ptr->has_same_characteristics(current_image_estimate))
    error("QuadraticBowsherPrior: kappa image has not the same index range as the reconstructed image\n");

  const int min_z = current_image_estimate.get_min_index();   
  const int max_z = current_image_estimate.get_max_index();   
  for (int z=min_z; z<=max_z; z++)  
    {  
      
      const int min_y = current_image_estimate[z].get_min_index();  
      const int max_y = current_image_estimate[z].get_max_index();  
       
      for (int y=min_y;y<= max_y;y++)  
        {     
           
          const int min_x = current_image_estimate[z][y].get_min_index(); 
          const int max_x = current_image_estimate[z][y].get_max_index();   
          for (int x=min_x;x<= max_x;x++) 
            { 
	      const int min_dz = max(weights[z][y][x].get_min_index(), min_z-z);  
	      const int max_dz = min(weights[z][y][x].get_max_index(), max_z-z); 
	      const int min_dy = max(weights[z][y][x][0].get_min_index(), min_y-y);  
	      const int max_dy = min(weights[z][y][x][0].get_max_index(), max_y-y);  
              const int min_dx = max(weights[z][y][x][0][0].get_min_index(), min_x-x); 
              const int max_dx = min(weights[z][y][x][0][0].get_max_index(), max_x-x); 
                
                elemT gradient = 0;
                for (int dz=min_dz;dz<=max_dz;++dz)
                  for (int dy=min_dy;dy<=max_dy;++dy)
                    for (int dx=min_dx;dx<=max_dx;++dx)
                      {
                        // 1 comes from omega = psi'(t)/t = 2*t/2t =1                  
                        elemT current =
                          weights[z][y][x][dz][dy][dx] *1;

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

template <typename elemT>
Succeeded 
QuadraticBowsherPrior<elemT>::
add_multiplication_with_approximate_Hessian(DiscretisedDensity<3,elemT>& output,
                                            const DiscretisedDensity<3,elemT>& input) const
{
  // TODO this function overlaps enormously with parabolic_surrogate_curvature
  // the only difference is that parabolic_surrogate_curvature uses input==1

  assert( output.has_same_characteristics(input));  
  if (this->penalisation_factor==0)
  {
    return Succeeded::yes;
  }
  
  DiscretisedDensityOnCartesianGrid<3,elemT>& output_cast =
    dynamic_cast<DiscretisedDensityOnCartesianGrid<3,elemT> &>(output);

  if (weights.get_length() ==0)
  {
    compute_weights(weights,  output_cast.get_index_range(), output_cast.get_grid_spacing(), this->only_2D, this->anatomical_image_ptr, this->nmax);
  }  
   
  const bool do_kappa = !is_null_ptr(kappa_ptr);
  
  if (do_kappa && !kappa_ptr->has_same_characteristics(input))
    error("QuadraticBowsherPrior: kappa image has not the same index range as the reconstructed image\n");

  const int min_z = output.get_min_index();   
  const int max_z = output.get_max_index();   
  for (int z=min_z; z<=max_z; z++)  
    {  
        
      const int min_y = output[z].get_min_index();  
      const int max_y = output[z].get_max_index();  
       
      for (int y=min_y;y<= max_y;y++)  
        {  
                       
          const int min_x = output[z][y].get_min_index(); 
          const int max_x = output[z][y].get_max_index();   

          for (int x=min_x;x<= max_x;x++) 
            { 
	      const int min_dz = max(weights[z][y][x].get_min_index(), min_z-z);  
	      const int max_dz = min(weights[z][y][x].get_max_index(), max_z-z);
	      const int min_dy = max(weights[z][y][x][0].get_min_index(), min_y-y);  
	      const int max_dy = min(weights[z][y][x][0].get_max_index(), max_y-y); 
              const int min_dx = max(weights[z][y][x][0][0].get_min_index(), min_x-x); 
              const int max_dx = min(weights[z][y][x][0][0].get_max_index(), max_x-x); 
                
                elemT result = 0;
                for (int dz=min_dz;dz<=max_dz;++dz)
                  for (int dy=min_dy;dy<=max_dy;++dy)
                    for (int dx=min_dx;dx<=max_dx;++dx)
                      {
                        elemT current =
                          weights[z][y][x][dz][dy][dx] * input[z+dz][y+dy][x+dx];

                         if (do_kappa)
                          current *= 
                            (*kappa_ptr)[z][y][x] * (*kappa_ptr)[z+dz][y+dy][x+dx];
                         result += current;
                      }
                
                output[z][y][x] += result * this->penalisation_factor;
            }              
        }
    }
  return Succeeded::yes;
}

#  ifdef _MSC_VER
// prevent warning message on reinstantiation, 
// note that we get a linking error if we don't have the explicit instantiation below
#  pragma warning(disable:4660)
#  endif


template class QuadraticBowsherPrior<float>;

END_NAMESPACE_STIR

