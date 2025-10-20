/*
    Copyright (C) 2025, University College London
    Copyright (C) 2025, University of Milano-Bicocca
    Copyright (C) 2000- 2011, Hammersmith Imanet Ltd

    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details.
*/
/*!
  \file
  \ingroup priors
  \brief Declaration of the stir::GibbsPenalty class

  \author Matteo Neel Colombo
  \author Kris Thielemans
  \author Sanida Mustafovic

*/

#ifndef __stir_recon_buildblock_GibbsPenalty_H__
#define __stir_recon_buildblock_GibbsPenalty_H__

#include "stir/recon_buildblock/GeneralisedPrior.h"
#include "stir/CartesianCoordinate3D.h"
#include "stir/Array.h"
#include "stir/DiscretisedDensity.h"
#include "stir/shared_ptr.h"
#include <string>

START_NAMESPACE_STIR

/*!
  \ingroup priors
  \brief A base class for Gibbs type penalties in the GeneralisedPrior hierarchy

  The prior is computed as follows:
  \f[
  f = \sum_{r,dr} w_{r,dr} \phi(\lambda_r , \lambda_{r+dr}) * \kappa_r * \kappa_{r+dr}
  \f]

  with gradient given by:
  \f[
  g_r = 2 * \sum_{dr} w_{r,dr} \phi'(\lambda_r , \lambda_{r+dr}) * \kappa_r * \kappa_{r+dr}
  \f]

  where \f$\lambda\f$ is the image and \f$r\f$ and \f$dr\f$ are indices and the sum
  is over the neighbourhood where the weights \f$w_{dr}\f$ are non-zero.

  The \f$\phi\f$ function is the potential function, which is provided via the template
  parameter PotentialFun. The potential function needs to be symmetric (\phi(x,y) = \phi(y,x)).
  Currently the potential function is implemented in the header of the derived classes
  (check GibbsQuadraticPrior.h or GibbsRelativeDifferencePrior.h to see examples).
  \par Potential Function Requirements
  The potential function class (PotentialT) must implement the following methods:

  - \c value(elemT val_center, elemT val_neigh, int z, int y, int x) - Returns the value of the potential function for the two
  voxel values
  - \c derivative_10(elemT val_center, elemT val_neigh, int z, int y, int x) - First derivative with respect to the center voxel
  - \c derivative_11(elemT val_center, elemT val_neigh, int z, int y, int x) - Second mixed derivative
  - \c derivative_20(elemT val_center, elemT val_neigh, int z, int y, int x) - Second derivative with respect to the center voxel
  - \c static bool is_convex() - Returns whether the potential function is convex
  - \c void initialise_keymap(KeyParser& parser) - Sets up parsing for any potential-specific parameters

  These methods should be declared with \c __host__ \c __device__ qualifiers,
  except for \c is_convex() and \c initialise_keymap() which are only used on the host.
  The coordinate parameters (z, y, x) may be used by the potential function for position-dependent behavior.
  Even if the potential has no parameters to parse, the \c initialise_keymap method must be implemented
  (possibly with an empty body).

  The \f$\kappa\f$ image can be used to have spatially-varying penalties such as in
  Jeff Fessler's papers. It should have identical dimensions to the image for which the
  penalty is computed. If \f$\kappa\f$ is not set, this class will effectively
  use 1 for all \f$\kappa\f$'s.

  By default, a 3x3 or 3x3x3 neigbourhood is used where the weights are set to
  x-voxel_size divided by the Euclidean distance between the points. Custom weights can be set
  using the method set_weights, the general form of the weights is NxNxN with N odd.

  \warning Currently only symmetric weights are supported (w_{i,j} = w_{j,i}).

  \par Parsing
  These are the keywords that can be used in all priors derived from GibbsPrior:
  \verbatim
  ; next defaults to 0, set to 1 for 2D inverse Euclidean weights, 0 for 3D
  only 2D:= 0
  ; next can be used to set weights explicitly. Needs to be a 3D array (of floats).
  ; weights:={{{0,1,0},{1,0,1},{0,1,0}}}
  ; use next parameter to specify an image with penalisation factors (a la Fessler)
  ; kappa filename:=
  ; use next parameter to get gradient images at every subiteration
  gradient filename prefix:=
  \endverbatim
*/
template <typename elemT, typename potentialT>
class GibbsPenalty : public GeneralisedPrior<DiscretisedDensity<3, elemT>>
{
private:
  typedef GeneralisedPrior<DiscretisedDensity<3, elemT>> base_type;

public:
  //! Default constructor.
  GibbsPenalty();

  //! Explicit Constructor with 2D/3D option and penalization factor.
  GibbsPenalty(const bool only_2D, float penalization_factor);

  //! Compute the value of the prior for the current image estimate.
  double compute_value(const DiscretisedDensity<3, elemT>& current_image_estimate) override;

  //! Compute the gradient of the prior for the current image estimate.
  void compute_gradient(DiscretisedDensity<3, elemT>& prior_gradient,
                        const DiscretisedDensity<3, elemT>& current_image_estimate) override;

  //! Compute the dot product of the prior gradient and an input image.
  double compute_gradient_times_input(const DiscretisedDensity<3, elemT>& input,
                                      const DiscretisedDensity<3, elemT>& current_image_estimate) override;

  //! Compute the Hessian row of the prior at (coords).
  void compute_Hessian(DiscretisedDensity<3, elemT>& prior_Hessian_for_single_densel,
                       const BasicCoordinate<3, int>& coords,
                       const DiscretisedDensity<3, elemT>& current_image_estimate) const override;

  //! Compute the diagonal of the Hessian matrix.
  void compute_Hessian_diagonal(DiscretisedDensity<3, elemT>& Hessian_diagonal,
                                const DiscretisedDensity<3, elemT>& current_estimate) const override;

  //! Accumulate Hessian times input image into output.
  void accumulate_Hessian_times_input(DiscretisedDensity<3, elemT>& output,
                                      const DiscretisedDensity<3, elemT>& current_estimate,
                                      const DiscretisedDensity<3, elemT>& input) const override;

  //! Get the current weights array.
  const Array<3, float>& get_weights() const;

  //! Set the weights array for the prior.
  virtual void set_weights(const Array<3, float>&);

  //! get current kappa image
  /*! \warning As this function returns a shared_ptr, this is dangerous. You should not
      modify the image by manipulating the image refered to by this pointer.
      Unpredictable results will occur.
  */
  shared_ptr<const DiscretisedDensity<3, elemT>> get_kappa_sptr() const;

  //! Set the kappa image (spatially-varying penalty factors).
  virtual void set_kappa_sptr(const shared_ptr<const DiscretisedDensity<3, elemT>>&);

  //! Set up the prior for a target image. Must be called before use.
  Succeeded set_up(shared_ptr<const DiscretisedDensity<3, elemT>> const& target_sptr) override;

  //! Getter method to retrieve the parsing name
  /*! Default implementation just uses get_registered_name(), and appends " Parameters". */
  virtual std::string get_parsing_name() const;

  //! Return whether the prior is convex or not
  bool is_convex() const override;

protected:
  //! @name Image and weight boundary indices
  //! @{
  CartesianCoordinate3D<int> image_dim;          //!< Image dimensions
  CartesianCoordinate3D<int> image_max_indices;  //!< Maximum image indices
  CartesianCoordinate3D<int> image_min_indices;  //!< Minimum image indices
  CartesianCoordinate3D<int> weight_max_indices; //!< Maximum weight indices
  CartesianCoordinate3D<int> weight_min_indices; //!< Minimum weight indices
  //! @}

  //! The weights for the neighbourhood
  Array<3, float> weights;

  //! can be set during parsing to restrict the weights to the 2D case
  bool only_2D;

  //! filename prefix for outputing the gradient whenever compute_gradient() is called.
  /*! An internal counter is used to keep track of the number of times the
     gradient is computed. The filename will be constructed by concatenating
     gradient_filename_prefix and the counter.
  */
  std::string gradient_filename_prefix;

  //! Filename for the \f$\kappa\f$ image that will be read by post_processing()
  std::string kappa_filename;

  //! Gibbs Potential Function
  potentialT potential;

  //! Compute default weights for the prior.
  void compute_default_weights(const CartesianCoordinate3D<float>& grid_spacing, bool only_2D);

  //! Check that the prior is ready to be used
  void check(DiscretisedDensity<3, elemT> const& current_image_estimate) const override;

  void set_defaults() override;
  void initialise_keymap() override;
  bool post_processing() override;

  //! The kappa image (spatially-varying penalty factors).
  shared_ptr<const DiscretisedDensity<3, elemT>> kappa_ptr;
};

END_NAMESPACE_STIR

#include "stir/recon_buildblock/GibbsPenalty.inl"

#endif
