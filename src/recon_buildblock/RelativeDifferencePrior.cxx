//
//
/*
    Copyright (C) 2000- 2019, Hammersmith Imanet Ltd
    Copyright (C) 2019- 2020, UCL
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup priors
  \brief  implementation of the stir::RelativeDifferencePrior class
    
  \author Kris Thielemans
  \author Sanida Mustafovic
  \author Robert Twyman

*/

#include "stir/recon_buildblock/RelativeDifferencePrior.h"
#include "stir/Succeeded.h"
#include "stir/DiscretisedDensityOnCartesianGrid.h"
#include "stir/IndexRange3D.h"
#include "stir/IO/write_to_file.h"
#include "stir/IO/read_from_file.h"
#include "stir/is_null_ptr.h"
#include "stir/info.h"
#include <algorithm>
using std::min;
using std::max;

/* Pretty horrible code because we don't have an iterator of neigbhourhoods yet
 */

START_NAMESPACE_STIR

template <typename elemT>
void 
RelativeDifferencePrior<elemT>::initialise_keymap()
{
  base_type::initialise_keymap();
  this->parser.add_start_key("Relative Difference Prior Parameters");
  this->parser.add_key("only 2D", &only_2D); 
  this->parser.add_key("kappa filename", &kappa_filename);
  this->parser.add_key("weights", &weights);
  this->parser.add_key("gradient filename prefix", &gradient_filename_prefix);
  this->parser.add_key("gamma value", &this->gamma);
  this->parser.add_key("epsilon value", &this->epsilon);
  this->parser.add_stop_key("END Relative Difference Prior Parameters");
}

template <typename elemT>
bool 
RelativeDifferencePrior<elemT>::post_processing()
{
  if (base_type::post_processing()==true)
    return true;
  if (kappa_filename.size() != 0)
    this->kappa_ptr = read_from_file<DiscretisedDensity<3,elemT> >(kappa_filename);

  bool warn_about_even_size = false;

  if (this->weights.size() ==0)
    {
      // will call compute_weights() to fill it in
    }
  else
    {
      if (!this->weights.is_regular())
        {
          warning("Sorry. RelativeDifferencePrior currently only supports regular arrays for the weights");
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
    warning("Parsing RelativeDifferencePrior: even number of weights occured in either x,y or z dimension.\n"
            "I'll (effectively) make this odd by appending a 0 at the end.");
  return false;

}

template <typename elemT>
Succeeded
RelativeDifferencePrior<elemT>::set_up (shared_ptr<DiscretisedDensity<3,elemT> > const& target_sptr)
{
  base_type::set_up(target_sptr);

  return Succeeded::yes;
}

template <typename elemT>
void RelativeDifferencePrior<elemT>::check(DiscretisedDensity<3,elemT> const& current_image_estimate) const
{
  // Do base-class check
  base_type::check(current_image_estimate);
}

template <typename elemT>
void
RelativeDifferencePrior<elemT>::set_defaults()
{
  base_type::set_defaults();
  this->only_2D = false;
  this->kappa_ptr.reset();  
  this->weights.recycle();
  this->gamma = 2;
  this->epsilon = 0.0;
}

template <>
const char * const 
RelativeDifferencePrior<float>::registered_name =
  "Relative Difference Prior";

template <typename elemT>
RelativeDifferencePrior<elemT>::RelativeDifferencePrior()
{
  set_defaults();
}

// Return the value of gamma - a RDP parameter
template <typename elemT>
float
RelativeDifferencePrior<elemT>::
get_gamma() const
{ return this->gamma; }

// Set the value of gamma - a RDP parameter
template <typename elemT>
void
RelativeDifferencePrior<elemT>::
set_gamma(float g)
{ this->gamma = g; }

// Return the value of epsilon - a RDP parameter
template <typename elemT>
float
RelativeDifferencePrior<elemT>::
get_epsilon() const
{ return this->epsilon; }

// Set the value of epsilon - a RDP parameter
template <typename elemT>
void
RelativeDifferencePrior<elemT>::
set_epsilon(float g)
{ this->epsilon = g; }


template <typename elemT>
RelativeDifferencePrior<elemT>::RelativeDifferencePrior(const bool only_2D_v, float penalisation_factor_v, float gamma_v, float epsilon_v)
  :  only_2D(only_2D_v)
{
  this->penalisation_factor = penalisation_factor_v;
  this->gamma = gamma_v;
  this->epsilon = epsilon_v;
}


  //! get penalty weights for the neighbourhood
template <typename elemT>
Array<3,float>  
RelativeDifferencePrior<elemT>::
get_weights() const
{ return this->weights; }

  //! set penalty weights for the neighbourhood
template <typename elemT>
void 
RelativeDifferencePrior<elemT>::
set_weights(const Array<3,float>& w)
{ this->weights = w; }

  //! get current kappa image
  /*! \warning As this function returns a shared_ptr, this is dangerous. You should not
      modify the image by manipulating the image referred to by this pointer.
      Unpredictable results will occur.
  */
template <typename elemT>
shared_ptr<DiscretisedDensity<3,elemT> >  
RelativeDifferencePrior<elemT>::
get_kappa_sptr() const
{ return this->kappa_ptr; }

  //! set kappa image
template <typename elemT>
void 
RelativeDifferencePrior<elemT>::
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
double
RelativeDifferencePrior<elemT>::
compute_value(const DiscretisedDensity<3,elemT> &current_image_estimate)
{
  if (this->penalisation_factor==0)
  {
    return 0.;
  }
  
  this->check(current_image_estimate);
  
  const DiscretisedDensityOnCartesianGrid<3,elemT>& current_image_cast =
    dynamic_cast< const DiscretisedDensityOnCartesianGrid<3,elemT> &>(current_image_estimate);
  
  if (this->weights.get_length() ==0)
  {
    compute_weights(this->weights, current_image_cast.get_grid_spacing(), this->only_2D);
  }
    
  const bool do_kappa = !is_null_ptr(kappa_ptr);
  
  if (do_kappa && !kappa_ptr->has_same_characteristics(current_image_estimate))
    error("RelativeDifferencePrior: kappa image has not the same index range as the reconstructed image\n");


  double result = 0.;
  const int min_z = current_image_estimate.get_min_index(); 
  const int max_z = current_image_estimate.get_max_index(); 
  for (int z=min_z; z<=max_z; z++)
    {
      const int min_dz = max(weights.get_min_index(), min_z-z);
      const int max_dz = min(weights.get_max_index(), max_z-z);
        
      const int min_y = current_image_estimate[z].get_min_index();
      const int max_y = current_image_estimate[z].get_max_index();

        for (int y=min_y;y<= max_y;y++)
          {
            const int min_dy = max(weights[0].get_min_index(), min_y-y);
            const int max_dy = min(weights[0].get_max_index(), max_y-y);            

            const int min_x = current_image_estimate[z][y].get_min_index(); 
            const int max_x = current_image_estimate[z][y].get_max_index(); 

            for (int x=min_x;x<= max_x;x++)       
              {
                const int min_dx = max(weights[0][0].get_min_index(), min_x-x);
                const int max_dx = min(weights[0][0].get_max_index(), max_x-x);
                
                for (int dz=min_dz;dz<=max_dz;++dz)
                  for (int dy=min_dy;dy<=max_dy;++dy)
                    for (int dx=min_dx;dx<=max_dx;++dx)
                      {
                        elemT current;
                        if (this->epsilon ==0.0 && current_image_estimate[z][y][x] == 0.0 && current_image_estimate[z+dz][y+dy][x+dx] == 0.0){
                          // handle the undefined nature of the function
                          current = 0.0;
                        } else {
                          current = weights[dz][dy][dx] * 0.5 *
                                  (pow(current_image_estimate[z][y][x]-current_image_estimate[z+dz][y+dy][x+dx],2)/
                                  (current_image_estimate[z][y][x]+current_image_estimate[z+dz][y+dy][x+dx]
                                  + this->gamma * abs(current_image_estimate[z][y][x]-current_image_estimate[z+dz][y+dy][x+dx]) + this->epsilon ));
                        }
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
RelativeDifferencePrior<elemT>::
compute_gradient(DiscretisedDensity<3,elemT>& prior_gradient, 
                 const DiscretisedDensity<3,elemT> &current_image_estimate)
{
  assert(  prior_gradient.has_same_characteristics(current_image_estimate));  
  if (this->penalisation_factor==0)
  {
    prior_gradient.fill(0);
    return;
  }

  this->check(current_image_estimate);
  
  
  const DiscretisedDensityOnCartesianGrid<3,elemT>& current_image_cast =
    dynamic_cast< const DiscretisedDensityOnCartesianGrid<3,elemT> &>(current_image_estimate);
  
  if (this->weights.get_length() ==0)
  {
    compute_weights(this->weights, current_image_cast.get_grid_spacing(), this->only_2D);
  }
 
 
  const bool do_kappa = !is_null_ptr(kappa_ptr);
  if (do_kappa && !kappa_ptr->has_same_characteristics(current_image_estimate))
    error("RelativeDifferencePrior: kappa image has not the same index range as the reconstructed image\n");

  const int min_z = current_image_estimate.get_min_index();  
  const int max_z = current_image_estimate.get_max_index();  
  for (int z=min_z; z<=max_z; z++) 
    { 
      const int min_dz = max(weights.get_min_index(), min_z-z); 
      const int max_dz = min(weights.get_max_index(), max_z-z); 
      
      const int min_y = current_image_estimate[z].get_min_index(); 
      const int max_y = current_image_estimate[z].get_max_index(); 
      
      for (int y=min_y;y<= max_y;y++) 
        { 
          const int min_dy = max(weights[0].get_min_index(), min_y-y); 
          const int max_dy = min(weights[0].get_max_index(), max_y-y);             
          
          const int min_x = current_image_estimate[z][y].get_min_index();
          const int max_x = current_image_estimate[z][y].get_max_index();  
          
          for (int x=min_x;x<= max_x;x++)
            {
              const int min_dx = max(weights[0][0].get_min_index(), min_x-x);
              const int max_dx = min(weights[0][0].get_max_index(), max_x-x);

                elemT gradient = 0;
                for (int dz=min_dz;dz<=max_dz;++dz)
                  for (int dy=min_dy;dy<=max_dy;++dy)
                    for (int dx=min_dx;dx<=max_dx;++dx)
                      {

                        elemT current;
                        if (this->epsilon ==0.0 && current_image_estimate[z][y][x] == 0.0 && current_image_estimate[z+dz][y+dy][x+dx] == 0.0){
                          // handle the undefined nature of the gradient
                          current = 0.0;
                        } else {
                            current = weights[dz][dy][dx] *
                                    (((current_image_estimate[z][y][x] - current_image_estimate[z+dz][y+dy][x+dx]) *
                                      (this->gamma * abs(current_image_estimate[z][y][x] - current_image_estimate[z+dz][y+dy][x+dx]) +
                                       current_image_estimate[z][y][x] + 3 * current_image_estimate[z+dz][y+dy][x+dx] + 2 * this->epsilon))/
                                     (square((current_image_estimate[z][y][x] + current_image_estimate[z+dz][y+dy][x+dx]) +
                                      this->gamma * abs(current_image_estimate[z][y][x] - current_image_estimate[z+dz][y+dy][x+dx]) + this->epsilon)));
                        }
                        if (do_kappa)
                          current *= (*kappa_ptr)[z][y][x] * (*kappa_ptr)[z+dz][y+dy][x+dx];

                        gradient += current;
                      }

                prior_gradient[z][y][x]= gradient * this->penalisation_factor;
              }              
          }
    }

  info(boost::format("Prior gradient max %1%, min %2%\n") % prior_gradient.find_max() % prior_gradient.find_min());

  static int count = 0;
  ++count;
  if (gradient_filename_prefix.size()>0)
    {
      char *filename = new char[gradient_filename_prefix.size()+100];
      sprintf(filename, "%s%d.v", gradient_filename_prefix.c_str(), count);
      write_to_file(filename, prior_gradient);
      delete[] filename;
    }
}

template <typename elemT>
Succeeded 
RelativeDifferencePrior<elemT>::
add_multiplication_with_approximate_Hessian(DiscretisedDensity<3,elemT>& output,
                                            const DiscretisedDensity<3,elemT>& input) const
{
   error("add_multiplication_with_approximate_Hessian()  is not implemented in Relative Difference Prior.");
  return Succeeded::no;
}

#  ifdef _MSC_VER
// prevent warning message on reinstantiation, 
// note that we get a linking error if we don't have the explicit instantiation below
#  pragma warning(disable:4660)
#  endif


template class RelativeDifferencePrior<float>;

END_NAMESPACE_STIR

