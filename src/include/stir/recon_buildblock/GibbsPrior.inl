//
//
/*
    Copyright (C) 2000- 2011, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup priors
  \brief  implementation of the stir::QuadraticPrior class

  \author Kris Thielemans
  \author Matteo Colombo

*/

#include "stir/recon_buildblock/GibbsPrior.h"
#include "stir/Succeeded.h"
#include "stir/DiscretisedDensityOnCartesianGrid.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/IndexRange3D.h"
#include "stir/IO/write_to_file.h"
#include "stir/BasicCoordinate.h"
#include "stir/IO/read_from_file.h"
#include "stir/is_null_ptr.h"
#include "stir/info.h"
#include "stir/warning.h"
#include "stir/error.h"
#include <algorithm>
#include <omp.h>
#include <omp.h>
using std::min;
using std::max;

START_NAMESPACE_STIR

template <typename elemT, typename potentialT>
void
GibbsPrior<elemT,potentialT>::initialise_keymap()
{
  base_type::initialise_keymap();
  this->parser.add_start_key("Gibbs Prior Parameters");
  this->parser.add_key("only 2D", &only_2D);
  this->parser.add_key("kappa filename", &kappa_filename);
  this->parser.add_key("weights", &weights);
  this->parser.add_key("gradient filename prefix", &gradient_filename_prefix);
  this->parser.add_stop_key("END Gibbs Prior Parameters");
}

template <typename elemT, typename potentialT>
bool
GibbsPrior<elemT,potentialT>::post_processing()
{
  if (base_type::post_processing() == true)
    return true;
  if (kappa_filename.size() != 0)
    this->kappa_ptr = read_from_file<DiscretisedDensity<3, elemT>>(kappa_filename);

  bool warn_about_even_size = false;

  if (this->weights.size() == 0)
    {
      // will call compute_weights() to fill it in
    }
  else
    {
      if (!this->weights.is_regular())
        {
          warning("Sorry. GibbsPrior currently only supports regular arrays for the weights");
          return true;
        }

      const unsigned int size_z = this->weights.size();
      if (size_z % 2 == 0)
        warn_about_even_size = true;
      const int min_index_z = -static_cast<int>(size_z / 2);
      this->weights.set_min_index(min_index_z);

      for (int z = min_index_z; z <= this->weights.get_max_index(); ++z)
        {
          const unsigned int size_y = this->weights[z].size();
          if (size_y % 2 == 0)
            warn_about_even_size = true;
          const int min_index_y = -static_cast<int>(size_y / 2);
          this->weights[z].set_min_index(min_index_y);
          for (int y = min_index_y; y <= this->weights[z].get_max_index(); ++y)
            {
              const unsigned int size_x = this->weights[z][y].size();
              if (size_x % 2 == 0)
                warn_about_even_size = true;
              const int min_index_x = -static_cast<int>(size_x / 2);
              this->weights[z][y].set_min_index(min_index_x);
            }
        }
    }

  if (warn_about_even_size)
    warning("Parsing GibbsPrior: even number of weights occured in either x,y or z dimension.\n"
            "I'll (effectively) make this odd by appending a 0 at the end.");
  return false;
}

template <typename elemT, typename potentialT>
Succeeded
GibbsPrior<elemT,potentialT>::set_up(shared_ptr<const DiscretisedDensity<3, elemT>> const& target_sptr)
{
  if (base_type::set_up(target_sptr) == Succeeded::no)
    return Succeeded::no;
  this->_already_set_up = false;
  auto& target_cast = dynamic_cast<const VoxelsOnCartesianGrid<elemT>&>(*target_sptr);
  
  // Set the default weights if not set
  if (weights.get_length() == 0)
    compute_default_weights(target_cast.get_voxel_size(), this->only_2D);
    
  auto sizes = target_cast.get_lengths();
  Image_dim = {sizes[1], sizes[2], sizes[3]};

  // Set the boundary of the image
  Image_min_indices = target_cast.get_min_indices();
  Image_max_indices = target_cast.get_max_indices();

  // Image_min_indices = {Im_min_indices[3], Im_min_indices[2], Im_min_indices[1]};
  // Image_max_indices = {Im_max_indices[3], Im_max_indices[2], Im_max_indices[1]};

  this->_already_set_up = true;
  return Succeeded::yes;
}

template <typename elemT, typename potentialT>
void
GibbsPrior<elemT,potentialT>::check(DiscretisedDensity<3, elemT> const& current_image_estimate) const
{
  // Do base-class check
  base_type::check(current_image_estimate);
  if (!is_null_ptr(this->kappa_ptr))
    {
      std::string explanation;
      if (!this->kappa_ptr->has_same_characteristics(current_image_estimate, explanation))
      //PROBLEMA REGISTERED NAME
        // error(std::string(registered_name)
        //       + ": kappa image does not have the same index range as the reconstructed image:" + explanation);
        error(": kappa image does not have the same index range as the reconstructed image:" + explanation);
    }
}

template <typename elemT, typename potentialT>
void
GibbsPrior<elemT,potentialT>::set_defaults()
{
  base_type::set_defaults();
  this->only_2D = false;
  this->kappa_ptr.reset();
  this->weights.recycle();
}

template <typename elemT, typename PotentialT>
GibbsPrior<elemT,PotentialT>::GibbsPrior()
{
  set_defaults();
}

template <typename elemT, typename PotentialT>
GibbsPrior<elemT,PotentialT>::GibbsPrior(const bool only_2D_v, float penalisation_factor_v)
    : only_2D(only_2D_v)
{
  this->penalisation_factor = penalisation_factor_v;
}

template <typename elemT, typename PotentialT>
bool
GibbsPrior<elemT,PotentialT>::is_convex() const
{
  return true;
}

// initialise to 1/Euclidean distance
template <typename elemT, typename PotentialT>
void
GibbsPrior<elemT,PotentialT>::compute_default_weights(const CartesianCoordinate3D<float>& grid_spacing, bool only_2D)
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
  weights = Array<3, float>(IndexRange3D(min_dz, max_dz, -1, 1, -1, 1));
  for (int z = min_dz; z <= max_dz; ++z)
    for (int y = -1; y <= 1; ++y)
      for (int x = -1; x <= 1; ++x)
        {
          if (z == 0 && y == 0 && x == 0)
            weights[0][0][0] = 0;
          else
            {
              weights[z][y][x]
                  = grid_spacing.x()
                    / sqrt(square(x * grid_spacing.x()) + square(y * grid_spacing.y()) + square(z * grid_spacing.z()));
            }
        }
  // Set the boundary of the weights
  weight_max_indices.z() = this->weights.get_max_index();
  weight_max_indices.y() = this->weights[0].get_max_index();
  weight_max_indices.x() = this->weights[0][0].get_max_index();

  weight_min_indices.z() = this->weights.get_min_index();
  weight_min_indices.y() = this->weights[0].get_min_index();
  weight_min_indices.x() = this->weights[0][0].get_min_index();
}

//! get penalty weights for the neigbourhood
template <typename elemT, typename PotentialT>
const Array<3, float>&
GibbsPrior<elemT,PotentialT>::get_weights() const
{
  return this->weights;
}

//! set penalty weights for the neigbourhood
template <typename elemT, typename PotentialT>
void
GibbsPrior<elemT,PotentialT>::set_weights(const Array<3, float>& w)
{
  this->weights = w;
  
  weight_max_indices.z() = w.get_max_index();
  weight_max_indices.y() = w[0].get_max_index();
  weight_max_indices.x() = w[0][0].get_max_index();

  weight_min_indices.z() = w.get_min_index();
  weight_min_indices.y() = w[0].get_min_index();
  weight_min_indices.x() = w[0][0].get_min_index();
} 

//! get current kappa image
/*! \warning As this function returns a shared_ptr, this is dangerous. You should not
    modify the image by manipulating the image refered to by this pointer.
    Unpredictable results will occur.
*/
template <typename elemT, typename PotentialT>
shared_ptr<const DiscretisedDensity<3, elemT>>
GibbsPrior<elemT,PotentialT>::get_kappa_sptr() const
{
  return this->kappa_ptr;
}

//! set kappa image
template <typename elemT, typename PotentialT>
void
GibbsPrior<elemT,PotentialT>::set_kappa_sptr(const shared_ptr<const DiscretisedDensity<3, elemT>>& k)
{
  this->kappa_ptr = k;
}

template <typename elemT, typename PotentialT>
double
GibbsPrior<elemT,PotentialT>::compute_value(const DiscretisedDensity<3, elemT>& current_image_estimate)
{
  //Preliminary Checks
  this->check(current_image_estimate);
  if (this->_already_set_up == false)
    error("GibbsPrior: set_up has not been called");
  if (this->penalisation_factor == 0)
    return 0.;
    
  const bool do_kappa = !is_null_ptr(kappa_ptr);

  double result = 0.0;
  #pragma omp parallel for reduction(+:result) 
  for (int z = Image_min_indices.z(); z <= Image_max_indices.z(); ++z)
    for (int y = Image_min_indices.y(); y <= Image_max_indices.y(); ++y)
      for (int x = Image_min_indices.x(); x <= Image_max_indices.x(); ++x)
      {

        const int min_dz = max(weight_min_indices.z(), Image_min_indices.z() - z);
        const int max_dz = min(weight_max_indices.z(), Image_max_indices.z() - z);
        const int min_dy = max(weight_min_indices.y(), Image_min_indices.y() - y);
        const int max_dy = min(weight_max_indices.y(), Image_max_indices.y() - y);
        const int min_dx = max(weight_min_indices.x(), Image_min_indices.x() - x);
        const int max_dx = min(weight_max_indices.x(), Image_max_indices.x() - x);
        //Periodic boundary on x and y tried, working but slower.
        const elemT val_center = current_image_estimate[z][y][x];
        
       
        for (int dz = min_dz; dz <= max_dz; ++dz)
          for (int dy = min_dy; dy <= max_dy; ++dy)
            for (int dx = min_dx; dx <= max_dx; ++dx)
              {
                  const elemT val_neigh = current_image_estimate[z + dz][y + dy][x + dx];
                  double current =  weights[dz][dy][dx] *
                                    this->potential.value(val_center, val_neigh, z, y, x);

                  if (do_kappa)
                      current *= (*kappa_ptr)[z][y][x] * (*kappa_ptr)[z + dz][y + dy][x + dx];

                  result += current;
              }
      }
        

  return result * this->penalisation_factor;
}

template <typename elemT, typename PotentialT>
void
GibbsPrior<elemT,PotentialT>::compute_gradient(DiscretisedDensity<3, elemT>& prior_gradient,
                                        const DiscretisedDensity<3, elemT>& current_image_estimate)
{
    //Preliminary Checks
  assert(prior_gradient.has_same_characteristics(current_image_estimate));
  this->check(current_image_estimate);
  if (this->_already_set_up == false)
    error("GibbsPrior: set_up has not been called");
  if (this->penalisation_factor == 0)
    {
      prior_gradient.fill(0);
      return;
    }
    
  const bool do_kappa = !is_null_ptr(kappa_ptr);

  #pragma omp parallel for
  for (int z = Image_min_indices.z(); z <= Image_max_indices.z(); ++z)
    for (int y = Image_min_indices.y(); y <= Image_max_indices.y(); ++y)
      for (int x = Image_min_indices.x(); x <= Image_max_indices.x(); ++x)
      {
        const int min_dz = max(weight_min_indices.z(), Image_min_indices.z() - z);
        const int max_dz = min(weight_max_indices.z(), Image_max_indices.z() - z);
        const int min_dy = max(weight_min_indices.y(), Image_min_indices.y() - y);
        const int max_dy = min(weight_max_indices.y(), Image_max_indices.y() - y);
        const int min_dx = max(weight_min_indices.x(), Image_min_indices.x() - x);
        const int max_dx = min(weight_max_indices.x(), Image_max_indices.x() - x);
        const elemT val_center = current_image_estimate[z][y][x];

        double gradient = 0.;
        for (int dz = min_dz; dz <= max_dz; ++dz)
          for (int dy = min_dy; dy <= max_dy; ++dy)
            for (int dx = min_dx; dx <= max_dx; ++dx)
              {
                if((dx ==0) && (dy == 0) && (dz == 0))
                  continue; 
                const elemT val_neigh = current_image_estimate[z + dz][y + dy][x + dx];
                double current = weights[dz][dy][dx]*
                                 this->potential.derivative_10(val_center, val_neigh, z, y, x);
                if (do_kappa)                                         
                  current *= (*kappa_ptr)[z][y][x] * (*kappa_ptr)[z + dz][y + dy][x + dx];
                gradient += current;
              }
        prior_gradient[z][y][x] = 2 * static_cast<elemT>(gradient * this->penalisation_factor);
      }
}

template <typename elemT, typename PotentialT>
double
GibbsPrior<elemT,PotentialT>::compute_gradient_times_input(const DiscretisedDensity<3, elemT>& input,
                                        const DiscretisedDensity<3, elemT>& current_image_estimate)
{
  //Preliminary Checks
  assert(input.has_same_characteristics(current_image_estimate));
  this->check(current_image_estimate);
  if (this->_already_set_up == false)
    error("GibbsPrior: set_up has not been called");
  if (this->penalisation_factor == 0)
    {
      return 0.0;
    }
    
  const bool do_kappa = !is_null_ptr(kappa_ptr);
  
  
  double result = 0.0;
  #pragma omp parallel for reduction(+:result)
  for (int z = Image_min_indices.z(); z <= Image_max_indices.z(); ++z)
    for (int y = Image_min_indices.y(); y <= Image_max_indices.y(); ++y)
      for (int x = Image_min_indices.x(); x <= Image_max_indices.x(); ++x)
      {
        const int min_dz = std::max(weight_min_indices.z(), Image_min_indices.z() - z);
        const int max_dz = std::min(weight_max_indices.z(), Image_max_indices.z() - z);
        const int min_dy = std::max(weight_min_indices.y(), Image_min_indices.y() - y);
        const int max_dy = std::min(weight_max_indices.y(), Image_max_indices.y() - y);
        const int min_dx = std::max(weight_min_indices.x(), Image_min_indices.x() - x);
        const int max_dx = std::min(weight_max_indices.x(), Image_max_indices.x() - x);
        const elemT val_center = current_image_estimate[z][y][x];

        double gradient = 0.0;
        for (int dz = min_dz; dz <= max_dz; ++dz)
          for (int dy = min_dy; dy <= max_dy; ++dy)
            for (int dx = min_dx; dx <= max_dx; ++dx)
              {
                if((dx ==0) && (dy == 0) && (dz == 0))
                  continue; 
                const elemT val_neigh = current_image_estimate[z + dz][y + dy][x + dx];
                double current = weights[dz][dy][dx]*
                                 this->potential.derivative_10(val_center, val_neigh, z, y, x);
                if (do_kappa)                                         
                  current *= (*kappa_ptr)[z][y][x] * (*kappa_ptr)[z + dz][y + dy][x + dx];
                gradient += current;
              }
        result += 2 * gradient * input[z][y][x];
      }
  return result * this->penalisation_factor;
}


template <typename elemT, typename PotentialT>
void
GibbsPrior<elemT,PotentialT>::compute_Hessian(DiscretisedDensity<3, elemT>& prior_Hessian_for_single_densel,
                                       const BasicCoordinate<3, int>& coords,
                                       const DiscretisedDensity<3, elemT>& current_image_estimate) const
{
  assert(prior_Hessian_for_single_densel.has_same_characteristics(current_image_estimate));
  prior_Hessian_for_single_densel.fill(0);
  if (this->penalisation_factor == 0)
    {
      return;
    }

  this->check(current_image_estimate);



  DiscretisedDensityOnCartesianGrid<3, elemT>& prior_Hessian_for_single_densel_cast
      = dynamic_cast<DiscretisedDensityOnCartesianGrid<3, elemT>&>(prior_Hessian_for_single_densel);


  const bool do_kappa = !is_null_ptr(kappa_ptr);

  const int z = coords[1];
  const int y = coords[2];
  const int x = coords[3];
  
  const int min_dz = max(weights.get_min_index(), prior_Hessian_for_single_densel.get_min_index() - z);
  const int max_dz = min(weights.get_max_index(), prior_Hessian_for_single_densel.get_max_index() - z);

  const int min_dy = max(weights[0].get_min_index(), prior_Hessian_for_single_densel[z].get_min_index() - y);
  const int max_dy = min(weights[0].get_max_index(), prior_Hessian_for_single_densel[z].get_max_index() - y);

  const int min_dx = max(weights[0][0].get_min_index(), prior_Hessian_for_single_densel[z][y].get_min_index() - x);
  const int max_dx = min(weights[0][0].get_max_index(), prior_Hessian_for_single_densel[z][y].get_max_index() - x);

  for (int dz = min_dz; dz <= max_dz; ++dz)
    for (int dy = min_dy; dy <= max_dy; ++dy)
      for (int dx = min_dx; dx <= max_dx; ++dx)
        {
          elemT current = 0.0;
          if (dz == 0 && dy == 0 && dx == 0)
            {
              // The j == k case (diagonal Hessian element), which is a sum over the neighbourhood.
              for (int ddz = min_dz; ddz <= max_dz; ++ddz)
                for (int ddy = min_dy; ddy <= max_dy; ++ddy)
                  for (int ddx = min_dx; ddx <= max_dx; ++ddx)
                    {
                      elemT diagonal_current
                          = weights[ddz][ddy][ddx]
                            * this->potential.derivative_20(current_image_estimate[z][y][x],
                                                                       current_image_estimate[z + dz][y + dy][x + dx],
                                                                       z, y, x);
                      if (do_kappa)
                        diagonal_current *= (*kappa_ptr)[z][y][x] * (*kappa_ptr)[z + ddz][y + ddy][x + ddx];
                      current += diagonal_current;
                    }
            }
          else
            {
              // The j != k vases (off-diagonal Hessian elements)
              current = weights[dz][dy][dx]
                        * this->potential.derivative_11(current_image_estimate[z][y][x],
                                                                       current_image_estimate[z + dz][y + dy][x + dx],
                                                                       z, y, x);
              if (do_kappa)
                current *= (*kappa_ptr)[z][y][x] * (*kappa_ptr)[z + dz][y + dy][x + dx];
            }
          prior_Hessian_for_single_densel_cast[z + dz][y + dy][x + dx] = +current * this->penalisation_factor;
        }
}

template <typename elemT, typename PotentialT>
void
GibbsPrior<elemT,PotentialT>::compute_Hessian_diagonal(DiscretisedDensity<3, elemT>& Hessian_diagonal,
                                            const DiscretisedDensity<3, elemT>& current_image_estimate) const
{
  //Preliminary Checks
  assert(Hessian_diagonal.has_same_characteristics(current_image_estimate));
  this->check(current_image_estimate);
  if (this->_already_set_up == false)
    error("GibbsPrior: set_up has not been called");
  if (this->penalisation_factor == 0)
    {
      Hessian_diagonal.fill(0);
      return;
    }

  const bool do_kappa = !is_null_ptr(kappa_ptr);

  #pragma omp parallel for
  for (int z = Image_min_indices.z(); z <= Image_max_indices.z(); ++z)
    for (int y = Image_min_indices.y(); y <= Image_max_indices.y(); ++y)
      for (int x = Image_min_indices.x(); x <= Image_max_indices.x(); ++x)
      {
        const int min_dz = max(weight_min_indices.z(), Image_min_indices.z() - z);
        const int max_dz = min(weight_max_indices.z(), Image_max_indices.z() - z);
        const int min_dy = max(weight_min_indices.y(), Image_min_indices.y() - y);
        const int max_dy = min(weight_max_indices.y(), Image_max_indices.y() - y);
        const int min_dx = max(weight_min_indices.x(), Image_min_indices.x() - x);
        const int max_dx = min(weight_max_indices.x(), Image_max_indices.x() - x);
        const elemT val_center = current_image_estimate[z][y][x];

        double Hessian_diag_element = 0.;
        for (int dz = min_dz; dz <= max_dz; ++dz)
          for (int dy = min_dy; dy <= max_dy; ++dy)
            for (int dx = min_dx; dx <= max_dx; ++dx)
              {
                if((dx ==0) && (dy == 0) && (dz == 0))
                  continue; 
                const elemT val_neigh = current_image_estimate[z + dz][y + dy][x + dx];
                double current = weights[dz][dy][dx]*
                                 this->potential.derivative_20(val_center, val_neigh, z, y, x);
                if (do_kappa)                                         
                  current *= (*kappa_ptr)[z][y][x] * (*kappa_ptr)[z + dz][y + dy][x + dx];
                Hessian_diag_element += current;
              }
        Hessian_diagonal[z][y][x] = 2 * static_cast<elemT>(Hessian_diag_element * this->penalisation_factor);
      }
}

template <typename elemT, typename PotentialT>
void
GibbsPrior<elemT,PotentialT>::accumulate_Hessian_times_input(DiscretisedDensity<3, elemT>& output,
                                                      const DiscretisedDensity<3, elemT>&  current_estimate,
                                                      const DiscretisedDensity<3, elemT>&  input) const
{
  // TODO this function overlaps enormously with parabolic_surrogate_curvature
  // the only difference is that parabolic_surrogate_curvature uses input==1

  assert(output.has_same_characteristics(input));
  if (this->penalisation_factor == 0)
    {
      return;
    }

  this->check(input);

  const bool do_kappa = !is_null_ptr(kappa_ptr);

  const int min_z = output.get_min_index();
  const int max_z = output.get_max_index();

  const int weight_min_z = weights.get_min_index();
  const int weight_max_z = weights.get_max_index();
  const int weight_min_y = weights[0].get_min_index();
  const int weight_max_y = weights[0].get_max_index();
  const int weight_min_x = weights[0][0].get_min_index();
  const int weight_max_x = weights[0][0].get_max_index();

  #pragma omp parallel for
  for (int z = min_z; z <= max_z; z++)
    {
      const int min_dz = max(weight_min_z, min_z - z);
      const int max_dz = min(weight_max_z, max_z - z);

      const int min_y = output[z].get_min_index();
      const int max_y = output[z].get_max_index();

      for (int y = min_y; y <= max_y; y++)
        {
          const int min_dy = max(weight_min_y, min_y - y);
          const int max_dy = min(weight_max_y, max_y - y);

          const int min_x = output[z][y].get_min_index();
          const int max_x = output[z][y].get_max_index();

          for (int x = min_x; x <= max_x; x++)
            {
              const int min_dx = max(weight_min_x, min_x - x);
              const int max_dx = min(weight_max_x, max_x - x);

              // At this point, we have j = [z][y][x]
              // The next for loops will have k = [z+dz][y+dy][x+dx]
              // The following computes
              //[H y]_j =
              //      \sum_{k\in N_j} w_{(j,k)} f''_{d}(x_j,x_k) y_j +
              //      \sum_{(i \in N_j) \ne j} w_{(j,i)} f''_{od}(x_j, x_i) y_i
              // Note the condition in the second sum that i is not equal to j

              elemT result = 0;
              for (int dz = min_dz; dz <= max_dz; ++dz)
              for (int dy = min_dy; dy <= max_dy; ++dy)
              for (int dx = min_dx; dx <= max_dx; ++dx)
              {
                elemT current = weights[dz][dy][dx];
                if (current == elemT(0))
                  continue;
       
                if ( (dz == 0) && (dy == 0) && (dx == 0))
                  {
                    current *= this->potential.derivative_20(current_estimate[z][y][x],
                                                                  current_estimate[z][y][x],
                                                                  z, y, x)
                                * input[z][y][x];          
                  }
                else
                  {
                    // std::cout<<'of_diag'<<std::endl;
                    current *= (potential.derivative_20(current_estimate[z][y][x], current_estimate[z + dz][y + dy][x + dx], z, y, x)
                                    * input[z][y][x]
                                + potential.derivative_11(current_estimate[z][y][x], current_estimate[z + dz][y + dy][x + dx], z, y, x)
                                      * input[z + dz][y + dy][x + dx]);
                  }

                if (do_kappa)
                  current *= (*kappa_ptr)[z][y][x] * (*kappa_ptr)[z + dz][y + dy][x + dx];
                result += current;
        }

              output[z][y][x] += 2 *result * this->penalisation_factor;
            }
        }
    }
}

END_NAMESPACE_STIR
