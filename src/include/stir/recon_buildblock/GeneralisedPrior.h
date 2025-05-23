//
//
/*
    Copyright (C) 2000- 2007, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup priors
  \brief Declaration of class stir::GeneralisedPrior

  \author Kris Thielemans
  \author Sanida Mustafovic

*/

#ifndef __stir_recon_buildblock_GeneralisedPrior_H__
#define __stir_recon_buildblock_GeneralisedPrior_H__


#include "stir/RegisteredObject.h"
#include "stir/ParsingObject.h"

START_NAMESPACE_STIR

class Succeeded;

/*!
  \ingroup priors
  \brief
  A base class for 'generalised' priors, i.e. priors for which at least
  a 'gradient' is defined. 

  This class exists to accomodate FilterRootPrior. Otherwise we could
  just live with Prior as a base class.
*/
template <typename DataT>
class GeneralisedPrior: 
   public RegisteredObject<GeneralisedPrior<DataT> >
			 
{
public:
  
  inline GeneralisedPrior(); 

  //! compute the value of the function
  /*! For derived classes where this doesn't make sense, it's recommended to return 0.
   */
  virtual double
    compute_value(const DataT &current_estimate) = 0;

  //! This should compute the gradient of the log of the prior function at the \a current_estimate
  /*! The gradient is already multiplied with the penalisation_factor.
      \warning The derived class should overwrite any data in \a prior_gradient.
  */
  virtual void compute_gradient(DataT& prior_gradient, 
		   const DataT &current_estimate) =0; 

  //! This should compute the multiplication of the Hessian with a vector and add it to \a output
  /*! Default implementation just call error(). This function needs to be overridden by the
      derived class.
      This method assumes that the hessian of the prior is 1 and hence the function quadratic.
      Instead, accumulate_Hessian_times_input() should be used. This method remains for backwards comparability.
       \warning The derived class should accumulate in \a output.
  */
  virtual Succeeded 
    add_multiplication_with_approximate_Hessian(DataT& output,
						const DataT& input) const;

    //! This should compute the multiplication of the Hessian with a vector and add it to \a output
    /*! Default implementation just call error(). This function needs to be overridden by the
        derived class.
        \warning The derived class should accumulate in \a output.
    */
  virtual Succeeded
  accumulate_Hessian_times_input(DataT& output,
          const DataT& current_estimate,
          const DataT& input) const;


  inline float get_penalisation_factor() const;
  inline void set_penalisation_factor(float new_penalisation_factor);

  //! Has to be called before using this object
  virtual Succeeded 
    set_up(shared_ptr<const DataT> const& target_sptr);

protected:
  float penalisation_factor;
  //! sets value for penalisation factor
  /*! Has to be called by set_defaults in the leaf-class */
  virtual void set_defaults();
  //! sets key for penalisation factor
  /*! Has to be called by initialise_keymap in the leaf-class */
  virtual void initialise_keymap();

  //! Check that the prior is ready to be used
  virtual void check(DataT const& current_estimate) const;

  bool _already_set_up;
};

END_NAMESPACE_STIR

#include "stir/recon_buildblock/GeneralisedPrior.inl"

#endif
