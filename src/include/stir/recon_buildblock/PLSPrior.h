//
//
/*
    Copyright (C) 2000- 2011, Hammersmith Imanet Ltd
    This file is part of STIR.

    This file is free software; you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation; either version 2.1 of the License, or
    (at your option) any later version.

    This file is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    See STIR/LICENSE.txt for details.
*/
/*!
  \file
  \ingroup priors
  \brief Declaration of class stir::PLSPrior

  \author Daniel Deidda
  \author Tsai Yu Jung
*/


#ifndef __stir_recon_buildblock_PLSPrior_H__
#define __stir_recon_buildblock_PLSPrior_H__


#include "stir/RegisteredParsingObject.h"
#include "stir/recon_buildblock/PriorWithParabolicSurrogate.h"
#include "stir/Array.h"
#include "stir/DiscretisedDensity.h"
#include "stir/shared_ptr.h"
#include <string>

START_NAMESPACE_STIR


/*!
  \ingroup priors
  \brief
  A class in the GeneralisedPrior hierarchy. This implements athe  anatomical penalty function, Parallel Level Sets (PLS),
  proposed by Matthias J. Ehrhardt et. al in "PET Reconstruction With an Anatomical MRI
  Prior Using Parallel Level Sets", IEEE Trans. med. Imag., vol. 35, no. 9, Sept. 2016. Note that
  PLS becomes smoothed TV when no anatomical information is available.

  The gradient of the prior is computed as follows:

  \f[
  g_r = \sum_dr w_{dr} (\lambda_r - \lambda_{r+dr}) * \kappa_r * \kappa_{r+dr}
  \f]
  where \f$\lambda\f$ is the image and \f$r\f$ and \f$dr\f$ are indices and the sum
  is over the neighbourhood where the weights \f$w_{dr}\f$ are non-zero.

  The \f$\kappa\f$ image can be used to have spatially-varying penalties such as in
  Jeff Fessler's papers. It should have identical dimensions to the image for which the
  penalty is computed. If \f$\kappa\f$ is not set, this class will effectively
  use 1 for all \f$\kappa\f$'s.

  By default, a 3x3 or 3x3x3 neigbourhood is used where the weights are set to
  x-voxel_size divided by the Euclidean distance between the points.

  \par Parsing
  These are the keywords that can be used in addition to the ones in GeneralPrior.
  \verbatim
  PLS Prior Parameters:=
  ; next defaults to 0, set to 1 for 2D inverse Euclidean weights, 0 for 3D
  only 2D:= 0


  anatomical_filename:= file.hv; Image that provides anatomical information (i.e. CT or MR images). The
                   dimension should be the same as that of the emission image.

  scale_par :=    ; A parameter for preventing the division by zero problem. The value dependes
                   on the scale of the anatomical image.

  smooth_par :=   ; A parameter that controls the edge-preservation property of PLS. The value
                   depends on the scale of the emission image.
  beta     :=     ; The global strength of PLS to the whole objective function

  ; use next parameter to specify an image with penalisation factors (a la Fessler)
  ; see class documentation for more info
  ; kappa filename:=
  ; use next parameter to get gradient images at every subiteration
  ; see class documentation
  gradient filename prefix:=
  END PLS Prior Parameters:=
  \endverbatim


*/
template <typename elemT>
class PLSPrior:  public
                       RegisteredParsingObject< PLSPrior<elemT>,
                                                GeneralisedPrior<DiscretisedDensity<3,elemT> >,
                                                PriorWithParabolicSurrogate<DiscretisedDensity<3,elemT> >
                                              >
{
 private:
  typedef
    RegisteredParsingObject< PLSPrior<elemT>,
                             GeneralisedPrior<DiscretisedDensity<3,elemT> >,
                             PriorWithParabolicSurrogate<DiscretisedDensity<3,elemT> > >
    base_type;

 public:
  //! Name which will be used when parsing a GeneralisedPrior object
  static const char * const registered_name;

  //! Default constructor
  PLSPrior();

  //! Constructs it explicitly
  PLSPrior(const bool only_2D, float penalization_factor);

  virtual bool
    parabolic_surrogate_curvature_depends_on_argument() const
    { return false; }

  //! compute the component x, y or z of the image gradient using forward difference
  void compute_image_gradient_element(DiscretisedDensity<3,elemT> & image_gradient_elem,
                                      std::string direction,
                                      const DiscretisedDensity<3,elemT> & image );

  //! Normalize the gradient of the anatomical image (Eq. (5) of the paper)
  void compute_normalis_image_gradient(DiscretisedDensity<3, elemT> &norm_im_grad,
                                            const DiscretisedDensity<3,elemT> &image_grad_z,
                                            const DiscretisedDensity<3,elemT> &image_grad_y,
                                            const DiscretisedDensity<3,elemT> &image_grad_x);
  //! Inner product in Eq. (9) of the paper but also the penalty function.
  void compute_inner_product_and_penalty(DiscretisedDensity<3,elemT> &inner_product,
                                           DiscretisedDensity<3,elemT> &penalty,
                                         DiscretisedDensity<3,elemT> &pet_im_grad_z,
                                         DiscretisedDensity<3,elemT> &pet_im_grad_y,
                                         DiscretisedDensity<3,elemT> &pet_im_grad_x,
                        const DiscretisedDensity<3,elemT> &pet_image);
  //! compute the value of the function
  double
    compute_value(const DiscretisedDensity<3,elemT> &current_image_estimate);

  //! compute gradient
  void compute_gradient(DiscretisedDensity<3,elemT>& prior_gradient,
                        const DiscretisedDensity<3,elemT> &current_image_estimate);

  //! compute the parabolic surrogate for the prior
  /*! in the case of PLS priors this will just be the sum of weighting coefficients*/
  void parabolic_surrogate_curvature(DiscretisedDensity<3,elemT>& parabolic_surrogate_curvature,
                        const DiscretisedDensity<3,elemT> &current_image_estimate);

  //! compute Hessian
  void compute_Hessian(DiscretisedDensity<3,elemT>& prior_Hessian_for_single_densel,
                const BasicCoordinate<3,int>& coords,
                const DiscretisedDensity<3,elemT> &current_image_estimate);

  virtual Succeeded
    add_multiplication_with_approximate_Hessian(DiscretisedDensity<3,elemT>& output,
                                                const DiscretisedDensity<3,elemT>& input) const;

  //! get penalty weights for the neigbourhood
  Array<3,float> get_weights() const;

  //! set penalty weights for the neigbourhood
  void set_weights(const Array<3,float>&);

  //! get current kappa image
  /*! \warning As this function returns a shared_ptr, this is dangerous. You should not
      modify the image by manipulating the image refered to by this pointer.
      Unpredictable results will occur.
  */
  shared_ptr<DiscretisedDensity<3,elemT> > get_kappa_sptr() const;
  shared_ptr<DiscretisedDensity<3,elemT> > get_anat_grad_sptr(std::string direction) const;
  shared_ptr<DiscretisedDensity<3,elemT> > get_norm_sptr() const;

  //! set kappa image
  void set_kappa_sptr(const shared_ptr<DiscretisedDensity<3,elemT> >&);

  //! set anatomical pointers
  void set_anat_grad_sptr(const shared_ptr<DiscretisedDensity<3,elemT> >&, std::string);
  void set_anat_grad_norm_sptr(const shared_ptr<DiscretisedDensity<3,elemT> >&);

protected:
  //! can be set during parsing to restrict the weights to the 2D case
  bool only_2D;
  //! filename prefix for outputing the gradient whenever compute_gradient() is called.
  /*! An internal counter is used to keep track of the number of times the
     gradient is computed. The filename will be constructed by concatenating
     gradient_filename_prefix and the counter.
  */
  std::string gradient_filename_prefix;

  //! penalty weights
  /*!
     \todo This member is mutable at present because some const functions initialise it.
     That initialisation should be moved to a new set_up() function.
  */
  mutable Array<3,float> weights;
  //! Filename for the \f$\kappa\f$ image that will be read by post_processing()
  std::string kappa_filename;
  std::string anatomical_filename;

  double scale_par, smooth_par;

  virtual void set_defaults();
  virtual void initialise_keymap();
  virtual bool post_processing();
 private:
  shared_ptr<DiscretisedDensity<3,elemT> > anat_grad_x_sptr;
  shared_ptr<DiscretisedDensity<3,elemT> > anat_grad_y_sptr;
  shared_ptr<DiscretisedDensity<3,elemT> > anat_grad_z_sptr;
  shared_ptr<DiscretisedDensity<3,elemT> > anatomical_sptr;
  shared_ptr<DiscretisedDensity<3,elemT> > norm_sptr;
  shared_ptr<DiscretisedDensity<3,elemT> > kappa_ptr;
  };


END_NAMESPACE_STIR

#endif

