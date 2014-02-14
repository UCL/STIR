//
//
/*
    Copyright (C) 2006- 2012, Hammersmith Imanet Ltd
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
  \brief  implementation of the stir::ParametricQuadraticPrior class 
    
  \author Kris Thielemans
  \author Sanida Mustafovic
  \author Charalampos Tsoumpas

*/

#include "local/stir/recon_buildblock/ParametricQuadraticPrior.h"
#include "stir/Succeeded.h"
#include "stir/IO/OutputFileFormat.h"
#include "stir/IO/read_from_file.h"

START_NAMESPACE_STIR

template <typename TargetT>
void 
ParametricQuadraticPrior<TargetT>::initialise_keymap()
{
  base_type::initialise_keymap();
  this->parser.add_start_key("Quadratic Prior Parameters");
  this->parser.add_key("only 2D", &this->only_2D); 
  this->parser.add_key("kappa filename", &this->kappa_filename);
  this->parser.add_key("weights", &this->weights);
  this->parser.add_key("gradient filename prefix", &this->gradient_filename_prefix);
  this->parser.add_stop_key("END Quadratic Prior Parameters");
}

template <typename TargetT>
bool 
ParametricQuadraticPrior<TargetT>::post_processing()
{
  if (base_type::post_processing()==true)
    return true;
  if (kappa_filename.size() != 0)
    this->kappa_ptr = read_from_file<TargetT>(kappa_filename);
  if (this->weights.size() ==0)
    {
      // will call compute_weights() to fill it in
    }
  else
    {
      for (unsigned int param_num=1; param_num<=TargetT::get_num_params(); ++param_num)
	{
	  this->_single_quadratic_priors[param_num].set_weights(this->get_weights()); // ChT: At the moment weights are treated equally.
	  //ChT: ToCheck
	  shared_ptr<typename TargetT::SingleDiscretisedDensityType> 
	    kappa_sptr(this->get_kappa_sptr()->construct_single_density(param_num).clone());
	  this->_single_quadratic_priors[param_num].set_kappa_sptr(kappa_sptr); 
	}
	// only_2D??
    }
  return false;
}

template <typename TargetT>
void
ParametricQuadraticPrior<TargetT>::set_defaults()
{
  base_type::set_defaults();
  this->only_2D = false;
  this->kappa_ptr.reset();  
  this->weights.recycle();
 // construct _single_quadratic_priors
  this->_single_quadratic_priors.resize(1,2);
}

template <>
const char * const 
ParametricQuadraticPrior<ParametricVoxelsOnCartesianGrid>::registered_name =
  "Quadratic";

template <typename TargetT>
ParametricQuadraticPrior<TargetT>::ParametricQuadraticPrior()
{
 // construct _single_quadratic_priors
  this->_single_quadratic_priors.resize(1,2);
  set_defaults();
}

template <typename TargetT>
ParametricQuadraticPrior<TargetT>::ParametricQuadraticPrior(const bool only_2D_v, float penalisation_factor_v)
  :  only_2D(only_2D_v)
{
  this->penalisation_factor = penalisation_factor_v; // should be able to ommit it
 // construct _single_quadratic_priors
  this->_single_quadratic_priors.resize(1,2);
   for (unsigned int param_num=1; param_num<=TargetT::get_num_params(); ++param_num)
	  this->_single_quadratic_priors[param_num].set_penalisation_factor(penalisation_factor_v);
   //What to do for the only 2D?
}

  //! get penalty weights for the neigbourhood
template <typename TargetT>
Array<3,float>  
ParametricQuadraticPrior<TargetT>::
get_weights() const
{ return this->weights; }

  //! set penalty weights for the neigbourhood
template <typename TargetT>
void 
ParametricQuadraticPrior<TargetT>::
set_weights(const Array<3,float>& w)
{ this->weights = w; }

  //! get current kappa image
  /*! \warning As this function returns a shared_ptr, this is dangerous. You should not
      modify the image by manipulating the image refered to by this pointer.
      Unpredictable results will occur.
  */
template <typename TargetT>
shared_ptr<TargetT >  
ParametricQuadraticPrior<TargetT>::
get_kappa_sptr() const
{ return this->kappa_ptr; }

  //! set kappa image
template <typename TargetT>
void 
ParametricQuadraticPrior<TargetT>::
set_kappa_sptr(const shared_ptr<TargetT >& k)
{ this->kappa_ptr = k; }

template <typename TargetT>
double
ParametricQuadraticPrior<TargetT>::
compute_value(const TargetT &current_image_estimate)
{
  double sum=0.; // At the moment I will have equal weights... so it is the sum (or the mean???) value of the two methods
  for (unsigned int param_num=1; param_num<=TargetT::get_num_params(); ++param_num)
    sum+=this->_single_quadratic_priors[param_num].compute_value(current_image_estimate.construct_single_density(param_num));  
  return sum;
}  

template <typename TargetT>
void 
ParametricQuadraticPrior<TargetT>::
compute_gradient(TargetT& prior_gradient, 
		 const TargetT &current_image_estimate)
{
  for (unsigned int param_num=1; param_num<=TargetT::get_num_params(); ++param_num)
    {
      typename TargetT::SingleDiscretisedDensityType single_density = prior_gradient.construct_single_density(param_num);
      this->_single_quadratic_priors[param_num].compute_gradient(single_density,current_image_estimate.construct_single_density(param_num));  
      prior_gradient.update_parametric_image(single_density,param_num);
    }
  if (gradient_filename_prefix.size()>0)
    {
      static int count = 0;
      ++count; // Maybe it will be usefult to add here a writing step, intialised by the parameter file, otherwise we will run out of space! 
      char *filename = new char[gradient_filename_prefix.size()+100];
      sprintf(filename, "%s%d.img", gradient_filename_prefix.c_str(), count);
      // This works only for ParametricVoxelsOnCartesianGrid and maybe for other ecat7 format files.    
      const Succeeded writing_succeeded=OutputFileFormat<TargetT>::default_sptr()->write_to_file(filename, prior_gradient); 
      delete[] filename;
    }
}

#if 0
template <typename TargetT>
void 
ParametricQuadraticPrior<TargetT>::
compute_Hessian(TargetT& prior_Hessian_for_single_densel, 
		const BasicCoordinate<3,int>& coords, const unsigned int input_param_num,
		const TargetT &current_image_estimate)
{
  for (unsigned int param_num=1; param_num<=TargetT::get_num_params(); ++param_num)
    {
      typename TargetT::SingleDiscretisedDensityType single_density = prior_Hessian_for_single_densel.construct_single_density(param_num);
      if (param_num == input_param_num)
	{	  
	  this->_single_quadratic_priors[param_num].compute_Hessian(single_density,
								    coords,current_image_estimate.construct_single_density(param_num));
	  prior_Hessian_for_single_densel.update_parametric_image(single_density.clone(),param_num);
	}
      else
	  prior_Hessian_for_single_densel.update_parametric_image(single_density.get_empty_copy(),param_num);
    }
}              
#endif

template <typename TargetT>
void 
ParametricQuadraticPrior<TargetT>::parabolic_surrogate_curvature(TargetT& parabolic_surrogate_curvature, 
			const TargetT &current_image_estimate)
{
  for (unsigned int param_num=1; param_num<=TargetT::get_num_params(); ++param_num)
    {
      typename TargetT::SingleDiscretisedDensityType single_density = parabolic_surrogate_curvature.construct_single_density(param_num);
      this->_single_quadratic_priors[param_num].parabolic_surrogate_curvature(single_density,current_image_estimate.construct_single_density(param_num));
      parabolic_surrogate_curvature.update_parametric_image(single_density,param_num);
    }
}

template <typename TargetT>
Succeeded 
ParametricQuadraticPrior<TargetT>::
add_multiplication_with_approximate_Hessian(TargetT& output,
					    const TargetT& input) const
{
  for (unsigned int param_num=1; param_num<=TargetT::get_num_params(); ++param_num)
    {
      typename TargetT::SingleDiscretisedDensityType single_density = output.construct_single_density(param_num);
      Succeeded if_success=this->_single_quadratic_priors[param_num].add_multiplication_with_approximate_Hessian(single_density,
														 input.construct_single_density(param_num));
      if(if_success==Succeeded::no)
	return if_success;
      else
	output.update_parametric_image(single_density,param_num);
    }
  return Succeeded::yes;
}


#  ifdef _MSC_VER
// prevent warning message on reinstantiation, 
// note that we get a linking error if we don't have the explicit instantiation below
#  pragma warning(disable:4660)
#  endif

template class ParametricQuadraticPrior<ParametricVoxelsOnCartesianGrid>; 

END_NAMESPACE_STIR


