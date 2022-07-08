//
//
/*
 Copyright (C) 2000- 2011, Hammersmith Imanet Ltd
 Copyright (C) 2016, 2020, UCL
 This file is part of STIR.

 SPDX-License-Identifier: Apache-2.0

 See STIR/LICENSE.txt for details
 */
/*!
 \file
 \ingroup priors
 \brief  implementation of the stir::LogcoshPrior class

 \author Kris Thielemans
 \author Sanida Mustafovic
 \author Yu-Jung Tsai
 \author Robert Twyman
 \author Zeljko Kereta
 */

#include "stir/recon_buildblock/LogcoshPrior.h"
#include "stir/Succeeded.h"
#include "stir/DiscretisedDensityOnCartesianGrid.h"
#include "stir/IndexRange3D.h"
#include "stir/IO/OutputFileFormat.h"
#include "stir/IO/read_from_file.h"
#include "stir/is_null_ptr.h"
#include "stir/info.h"
#include <algorithm>
using std::min;
using std::max;

START_NAMESPACE_STIR

template <typename elemT>
void
LogcoshPrior<elemT>::initialise_keymap()
{
  base_type::initialise_keymap();
  this->parser.add_start_key("Logcosh Prior Parameters");
  this->parser.add_key("only 2D", &only_2D);
  this->parser.add_key("scalar", &scalar);
  this->parser.add_key("kappa filename", &kappa_filename);
  this->parser.add_key("weights", &weights);
  this->parser.add_key("gradient filename prefix", &gradient_filename_prefix);
  this->parser.add_stop_key("END Logcosh Prior Parameters");
}

template <typename elemT>
bool
LogcoshPrior<elemT>::post_processing()
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
      warning("Sorry. LogcoshPrior currently only supports regular arrays for the weights");
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
    warning("Parsing LogcoshPrior: even number of weights occurred in either x,y or z dimension.\n"
            "I'll (effectively) make this odd by appending a 0 at the end.");
  return false;

}

template <typename elemT>
void
LogcoshPrior<elemT>::set_defaults()
{
  base_type::set_defaults();
  this->only_2D = false;
  this->scalar = 1.0;
  this->kappa_ptr.reset();
  this->weights.recycle();
}

template <>
const char * const
        LogcoshPrior<float>::registered_name =
        "Logcosh";

template <typename elemT>
LogcoshPrior<elemT>::LogcoshPrior()
{
  set_defaults();
}


template <typename elemT>
LogcoshPrior<elemT>::LogcoshPrior(const bool only_2D_v, float penalisation_factor_v)
        :  only_2D(only_2D_v)
{
  set_defaults();
  this->penalisation_factor = penalisation_factor_v;
}


template <typename elemT>
LogcoshPrior<elemT>::LogcoshPrior(const bool only_2D_v, float penalisation_factor_v, const float scalar_v)
        :  only_2D(only_2D_v), scalar(scalar_v)
{
  set_defaults();
  this->penalisation_factor = penalisation_factor_v;
  this->scalar = scalar_v;
}

template <typename elemT>
bool
LogcoshPrior<elemT>::
is_convex() const
{
  return true;
}

//! get penalty weights for the neighbourhood
template <typename elemT>
Array<3,float>
LogcoshPrior<elemT>::
get_weights() const
{ return this->weights; }

//! set penalty weights for the neighbourhood
template <typename elemT>
void
LogcoshPrior<elemT>::
set_weights(const Array<3,float>& w)
{ this->weights = w; }

//! get current kappa image
/*! \warning As this function returns a shared_ptr, this is dangerous. You should not
 modify the image by manipulating the image referred to by this pointer.
 Unpredictable results will occur.
 */
template <typename elemT>
shared_ptr<DiscretisedDensity<3,elemT> >
LogcoshPrior<elemT>::
get_kappa_sptr() const
{ return this->kappa_ptr; }

//! set kappa image
template <typename elemT>
void
LogcoshPrior<elemT>::
set_kappa_sptr(const shared_ptr<DiscretisedDensity<3,elemT> >& k)
{ this->kappa_ptr = k; }

//! Get the scalar value
template <typename elemT>
float
LogcoshPrior<elemT>::
get_scalar() const
{ return this->scalar; }

//! Set the scalar value
template <typename elemT>
void
LogcoshPrior<elemT>::
set_scalar(const float scalar_v)
{ scalar = scalar_v; }


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
LogcoshPrior<elemT>::
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
    compute_weights(this->weights, current_image_cast.get_grid_spacing(), this->only_2D);
  }

  const bool do_kappa = !is_null_ptr(kappa_ptr);

  if (do_kappa && !kappa_ptr->has_same_characteristics(current_image_estimate))
    error("LogcoshPrior: kappa image has not the same index range as the reconstructed image\n");

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

        /* formula:
         sum_dx,dy,dz
         weights[dz][dy][dx] *
         log(cosh(current_image_estimate[z][y][x] - current_image_estimate[z+dz][y+dy][x+dx])) *
         (*kappa_ptr)[z][y][x] * (*kappa_ptr)[z+dz][y+dy][x+dx];
         */
        for (int dz=min_dz;dz<=max_dz;++dz)
          for (int dy=min_dy;dy<=max_dy;++dy)
            for (int dx=min_dx;dx<=max_dx;++dx)
            {
              // 1/scalar^2 * log(cosh(x * scalar))
              elemT voxel_diff= current_image_estimate[z][y][x] - current_image_estimate[z+dz][y+dy][x+dx];
              elemT current = weights[dz][dy][dx] * 1/(this->scalar*this->scalar) * logcosh(this->scalar*voxel_diff);
              if (do_kappa)
                current *= (*kappa_ptr)[z][y][x] * (*kappa_ptr)[z+dz][y+dy][x+dx];
              result += static_cast<double>(current);
            }
      }
    }
  }
  return result * this->penalisation_factor/2.0;
}

template <typename elemT>
void
LogcoshPrior<elemT>::
compute_gradient(DiscretisedDensity<3,elemT>& prior_gradient,
                 const DiscretisedDensity<3,elemT> &current_image_estimate)
{
  assert(prior_gradient.has_same_characteristics(current_image_estimate));
  if (this->penalisation_factor==0)
  {
    prior_gradient.fill(0);
    return;
  }

  const DiscretisedDensityOnCartesianGrid<3,elemT>& current_image_cast =
          dynamic_cast< const DiscretisedDensityOnCartesianGrid<3,elemT> &>(current_image_estimate);

  if (this->weights.get_length() ==0)
  {
    compute_weights(this->weights, current_image_cast.get_grid_spacing(), this->only_2D);
  }

  const bool do_kappa = !is_null_ptr(kappa_ptr);
  if (do_kappa && !kappa_ptr->has_same_characteristics(current_image_estimate))
    error("LogcoshPrior: kappa image has not the same index range as the reconstructed image\n");

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
              // 1/scalar * tanh(x * scalar)
              elemT voxel_diff= current_image_estimate[z][y][x] - current_image_estimate[z+dz][y+dy][x+dx];
              elemT current = weights[dz][dy][dx] * (1/this->scalar) * tanh(this->scalar*voxel_diff);

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
    OutputFileFormat<DiscretisedDensity<3,elemT> >::default_sptr()->
            write_to_file(filename, prior_gradient);
    delete[] filename;
  }
}

template <typename elemT>
void
LogcoshPrior<elemT>::
compute_Hessian(DiscretisedDensity<3,elemT>& prior_Hessian_for_single_densel,
                const BasicCoordinate<3,int>& coords,
                const DiscretisedDensity<3,elemT> &current_image_estimate) const
{
  assert(  prior_Hessian_for_single_densel.has_same_characteristics(current_image_estimate));
  prior_Hessian_for_single_densel.fill(0);
  if (this->penalisation_factor==0)
  {
    return;
  }

  this->check(current_image_estimate);

  const DiscretisedDensityOnCartesianGrid<3,elemT>& current_image_cast =
          dynamic_cast< const DiscretisedDensityOnCartesianGrid<3,elemT> &>(current_image_estimate);

  DiscretisedDensityOnCartesianGrid<3,elemT>& prior_Hessian_for_single_densel_cast =
          dynamic_cast<DiscretisedDensityOnCartesianGrid<3,elemT> &>(prior_Hessian_for_single_densel);

  if (weights.get_length() ==0)
  {
    compute_weights(weights, current_image_cast.get_grid_spacing(), this->only_2D);
  }

  const bool do_kappa = !is_null_ptr(kappa_ptr);

  if (do_kappa && kappa_ptr->has_same_characteristics(current_image_estimate))
    error("LogcoshPrior: kappa image has not the same index range as the reconstructed image\n");

  const int z = coords[1];
  const int y = coords[2];
  const int x = coords[3];
  const int min_dz = max(weights.get_min_index(), prior_Hessian_for_single_densel.get_min_index()-z);
  const int max_dz = min(weights.get_max_index(), prior_Hessian_for_single_densel.get_max_index()-z);

  const int min_dy = max(weights[0].get_min_index(), prior_Hessian_for_single_densel[z].get_min_index()-y);
  const int max_dy = min(weights[0].get_max_index(), prior_Hessian_for_single_densel[z].get_max_index()-y);

  const int min_dx = max(weights[0][0].get_min_index(), prior_Hessian_for_single_densel[z][y].get_min_index()-x);
  const int max_dx = min(weights[0][0].get_max_index(), prior_Hessian_for_single_densel[z][y].get_max_index()-x);

  elemT diagonal = 0;
  for (int dz=min_dz;dz<=max_dz;++dz)
    for (int dy=min_dy;dy<=max_dy;++dy)
      for (int dx=min_dx;dx<=max_dx;++dx)
      {
        elemT current = 0.0;
        if (dz == 0 && dy == 0 && dx == 0)
        {
          // The j == k case (diagonal Hessian element), which is a sum over the neighbourhood.
          for (int ddz=min_dz;ddz<=max_dz;++ddz)
            for (int ddy=min_dy;ddy<=max_dy;++ddy)
              for (int ddx=min_dx;ddx<=max_dx;++ddx)
              {
                elemT diagonal_current = weights[ddz][ddy][ddx] *
                        derivative_20(current_image_estimate[z][y][x],
                                      current_image_estimate[z + ddz][y + ddy][x + ddx]);
                if (do_kappa)
                  diagonal_current *= (*kappa_ptr)[z][y][x] * (*kappa_ptr)[z+ddz][y+ddy][x+ddx];
                current += diagonal_current;
              }
        }
        else
        {
          // The j != k vases (off-diagonal Hessian elements)
          current = weights[dz][dy][dx] * derivative_11(current_image_estimate[z][y][x],
                                                        current_image_estimate[z + dz][y + dy][x + dx]);
          if (do_kappa)
            current *= (*kappa_ptr)[z][y][x] * (*kappa_ptr)[z+dz][y+dy][x+dx];
        }
        prior_Hessian_for_single_densel_cast[z+dz][y+dy][x+dx] = + current*this->penalisation_factor;
      }
}


template <typename elemT>
void
LogcoshPrior<elemT>::parabolic_surrogate_curvature(DiscretisedDensity<3,elemT>& parabolic_surrogate_curvature,
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
    compute_weights(weights, current_image_cast.get_grid_spacing(), this->only_2D);
  }

  const bool do_kappa = !is_null_ptr(kappa_ptr);

  if (do_kappa && !kappa_ptr->has_same_characteristics(current_image_estimate))
    error("LogcoshPrior: kappa image has not the same index range as the reconstructed image\n");

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
              // psi'(t)/t = tanh/t
              elemT voxel_diff=current_image_estimate[z][y][x] - current_image_estimate[z+dz][y+dy][x+dx];
              elemT current = weights[dz][dy][dx] * surrogate(voxel_diff, this->scalar);

              if (do_kappa)
                current *= (*kappa_ptr)[z][y][x] * (*kappa_ptr)[z+dz][y+dy][x+dx];

              gradient += current;
            }
        parabolic_surrogate_curvature[z][y][x]= gradient * this->penalisation_factor;
      }
    }
  }
  info(boost::format("parabolic_surrogate_curvature max %1%, min %2%\n") % parabolic_surrogate_curvature.find_max() % parabolic_surrogate_curvature.find_min());
}

template <typename elemT>
void
LogcoshPrior<elemT>::
accumulate_Hessian_times_input(DiscretisedDensity<3,elemT>& output,
                               const DiscretisedDensity<3,elemT>& current_estimate,
                               const DiscretisedDensity<3,elemT>& input) const
{
  // TODO this function overlaps enormously with parabolic_surrogate_curvature
  // the only difference is that parabolic_surrogate_curvature uses input==1

  assert( output.has_same_characteristics(input));
  if (this->penalisation_factor==0)
  {
    return;
  }

  DiscretisedDensityOnCartesianGrid<3,elemT>& output_cast =
          dynamic_cast<DiscretisedDensityOnCartesianGrid<3,elemT> &>(output);

  if (weights.get_length() ==0)
  {
    compute_weights(weights, output_cast.get_grid_spacing(), this->only_2D);
  }

  const bool do_kappa = !is_null_ptr(kappa_ptr);

  if (do_kappa && !kappa_ptr->has_same_characteristics(input))
    error("LogcoshPrior: kappa image has not the same index range as the reconstructed image\n");

  const int min_z = output.get_min_index();
  const int max_z = output.get_max_index();
  for (int z=min_z; z<=max_z; z++)
  {
    const int min_dz = max(weights.get_min_index(), min_z-z);
    const int max_dz = min(weights.get_max_index(), max_z-z);

    const int min_y = output[z].get_min_index();
    const int max_y = output[z].get_max_index();

    for (int y=min_y;y<= max_y;y++)
    {
      const int min_dy = max(weights[0].get_min_index(), min_y-y);
      const int max_dy = min(weights[0].get_max_index(), max_y-y);

      const int min_x = output[z][y].get_min_index();
      const int max_x = output[z][y].get_max_index();

      for (int x=min_x;x<= max_x;x++)
      {
        const int min_dx = max(weights[0][0].get_min_index(), min_x-x);
        const int max_dx = min(weights[0][0].get_max_index(), max_x-x);

        elemT result = 0;
        for (int dz=min_dz;dz<=max_dz;++dz)
          for (int dy=min_dy;dy<=max_dy;++dy)
            for (int dx=min_dx;dx<=max_dx;++dx)
            {
              elemT current = weights[dz][dy][dx];
              if (dz == dy == dz == 0) {
                // The j == k case
                current *= derivative_20(current_estimate[z][y][x],
                                         current_estimate[z + dz][y + dy][x + dx]) * input[z][y][x];
              } else {
                current *= (derivative_20(current_estimate[z][y][x],
                                          current_estimate[z + dz][y + dy][x + dx]) * input[z][y][x] +
                        derivative_11(current_estimate[z][y][x],
                                      current_estimate[z + dz][y + dy][x + dx]) *
                            input[z + dz][y + dy][x + dx]);
              }

              if (do_kappa)
                current *= (*kappa_ptr)[z][y][x] * (*kappa_ptr)[z+dz][y+dy][x+dx];

              result += current;
            }

        output[z][y][x] += result * this->penalisation_factor;
      }
    }
  }
}

template <typename elemT>
elemT
LogcoshPrior<elemT>::
derivative_20(const elemT x_j, const elemT x_k) const
{
  return square((1/ cosh((x_j - x_k) * scalar)));
}

template <typename elemT>
elemT
LogcoshPrior<elemT>::
derivative_11(const elemT x_j, const elemT x_k) const
{
  return -derivative_20(x_j, x_k);
}

#  ifdef _MSC_VER
    // prevent warning message on reinstantiation,
// note that we get a linking error if we don't have the explicit instantiation below
#  pragma warning(disable:4660)
#  endif

    template class LogcoshPrior<float>;

END_NAMESPACE_STIR
