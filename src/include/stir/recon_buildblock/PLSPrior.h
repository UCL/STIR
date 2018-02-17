//
//
/*
    Copyright (C) 2018 University of Leeds and University College of London

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
  \author Yu-Jung Tsai
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
  A class in the GeneralisedPrior hierarchy. This implements the  anatomical penalty function, Parallel Level Sets (PLS),
  proposed by Matthias J. Ehrhardt et. al in "PET Reconstruction With an Anatomical MRI
  Prior Using Parallel Level Sets", IEEE Trans. med. Imag., vol. 35, no. 9, Sept. 2016. Note that
  PLS becomes smoothed TV when an uniform anatomical image is provided.

  The prior is computed as follows:

  \f[
  \phi(f) = \sqrt{\alpha^2 + |\nabla v|^2 - {\langle\nabla f,\xi\rangle}^2}
  \f]
  where \f$ f \f$ is the PET image and \f$ \alpha \f$ is a parameter that controls the edge-preservation property of PLS.

  The \f$ \xi \f$ is the normalised gradient calculated as follows:

  \f[
  \xi = \frac{\nabla v}{\sqrt{|\nabla v|^2 + \eta^2}}
  \f]

  where \f$ v f$ is the anatomical image and \f$ \eta \f$ is a parameter for preventing the division by zero problem.


  A \f$\kappa\f$ image can be used to have spatially-varying penalties such as in
  Jeff Fessler's papers. It should have identical dimensions to the image for which the
  penalty is computed. If \f$\kappa\f$ is not set, this class will effectively
  use 1 for all \f$\kappa\f$'s.



  \par Parsing
  These are the keywords that can be used in addition to the ones in GeneralPrior.
  \verbatim
  PLS Prior Parameters:=
  ; next defaults to 0, set to 1 for 2D images, 0 for 3D
  only 2D:= 0


  anatomical_filename:= file.hv; Image that provides anatomical information (i.e. CT or MR image). The
                   dimension should be the same as that of the emission image.

  eta :=    ; A parameter for preventing the division by zero problem. The value dependes
                   on the scale of the anatomical image.

  alpha :=   ; A parameter that controls the edge-preservation property of PLS. The value
                   depends on the scale of the emission image.

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
                                                GeneralisedPrior<DiscretisedDensity<3,elemT> >
                                              >
{
 private:
  typedef
    RegisteredParsingObject< PLSPrior<elemT>,
                             GeneralisedPrior<DiscretisedDensity<3,elemT> >,
                             GeneralisedPrior<DiscretisedDensity<3,elemT> > >
    base_type;

 public:
  //! Name which will be used when parsing a GeneralisedPrior object
  static const char * const registered_name;

  //! Default constructor
  PLSPrior();

  //! Constructs it explicitly
  PLSPrior(const bool only_2D, float penalization_factor);


  //! compute the value of the function
  double
    compute_value(const DiscretisedDensity<3,elemT> &current_image_estimate);

  //! compute gradient
  void compute_gradient(DiscretisedDensity<3,elemT>& prior_gradient,
                        const DiscretisedDensity<3,elemT> &current_image_estimate);

  //! get current kappa image
  /*! \warning As this function returns a shared_ptr, this is dangerous. You should not
      modify the image by manipulating the image refered to by this pointer.
      Unpredictable results will occur.
  */
  shared_ptr<DiscretisedDensity<3,elemT> > get_kappa_sptr() const;
  shared_ptr<DiscretisedDensity<3,elemT> > get_anatomical_grad_sptr(int direction) const;
  shared_ptr<DiscretisedDensity<3,elemT> > get_norm_sptr() const;

  //!get eta and alpha parameters
  double get_eta() const;
  double get_alpha() const;

  //! set kappa image
  void set_kappa_sptr(const shared_ptr<DiscretisedDensity<3,elemT> >&);

  //! set/get anatomical pointers
  void set_anatomical_image_sptr(const shared_ptr<DiscretisedDensity<3,elemT> >&);
  shared_ptr<DiscretisedDensity<3,elemT> > get_anatomical_image_sptr() const;

protected:
  //! can be set during parsing to restrict the gradient calculation to the 2D case
  bool only_2D;
  //! filename prefix for outputing the gradient whenever compute_gradient() is called.
  /*! An internal counter is used to keep track of the number of times the
     gradient is computed. The filename will be constructed by concatenating
     gradient_filename_prefix and the counter.
  */
  std::string gradient_filename_prefix;

  //! Filename for the \f$\kappa\f$ image that will be read by post_processing()
  std::string kappa_filename;
  std::string anatomical_filename;

  double eta, alpha;

  virtual void set_defaults();
  virtual void initialise_keymap();

  //! the parsing will only override any exixting kappa-image or anatomical-image if the relevant keyword is present
  virtual bool post_processing();
 private:

    /*! \todo set the anatomical image to zero if not defined */
   virtual Succeeded set_up();

  //! compute the component x, y or z of the image gradient using forward difference
  static void compute_image_gradient_element(DiscretisedDensity<3,elemT> & image_gradient_elem,
                                      int direction,
                                      const DiscretisedDensity<3,elemT> & image );

  //! compute normalisation for the gradient of the anatomical image (Eq. (5) of the paper)
  void compute_normalisation_anatomical_gradient(DiscretisedDensity<3, elemT> &norm_im_grad,
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

  shared_ptr<DiscretisedDensity<3,elemT> > anatomical_grad_x_sptr;
  shared_ptr<DiscretisedDensity<3,elemT> > anatomical_grad_y_sptr;
  shared_ptr<DiscretisedDensity<3,elemT> > anatomical_grad_z_sptr;
  shared_ptr<DiscretisedDensity<3,elemT> > anatomical_sptr;
  shared_ptr<DiscretisedDensity<3,elemT> > norm_sptr;
  shared_ptr<DiscretisedDensity<3,elemT> > kappa_ptr;
  void set_anatomical_grad_sptr(const shared_ptr<DiscretisedDensity<3,elemT> >&, int);
  void set_anatomical_grad_norm_sptr(const shared_ptr<DiscretisedDensity<3,elemT> >&);
  void set_eta(const double&);
  void set_alpha(const double&);
  };


END_NAMESPACE_STIR

#endif

