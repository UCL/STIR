//
//
/*
 Copyright (C) 2000- 2011, Hammersmith Imanet Ltd
 Copyright (C) 2016, 2020, UCL
 This file is part of STIR.

 SPDX-License-Identifier: Apache-2.0

 See STIR/LICENSE.txt for details.
 */
/*!
 \file
 \ingroup priors
 \brief Declaration of class stir::LogcoshPrior

 \author Kris Thielemans
 \author Sanida Mustafovic
 \author Yu-Jung Tsai
 \author Robert Twyman
 \author Zeljko Kereta
 */


#ifndef __stir_recon_buildblock_LogcoshPrior_H__
#define __stir_recon_buildblock_LogcoshPrior_H__


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
 A class in the GeneralisedPrior hierarchy. This implements a logcosh Gibbs prior.

 The log-cosh function is given by:
  \f[
    f = \sum_{dr} w_{dr} 1/scalar^2 * log(cosh(\lambda_r - \lambda_{r+dr})) * \kappa_r * \kappa_{r+dr}))
 \f]

 where \f$\lambda\f$ is the image and \f$r\f$ and \f$dr\f$ are indices and the sum
 is over the neighbourhood where the weights \f$w_{dr}\f$ are non-zero.

 The \f$\kappa\f$ image can be used to have spatially-varying penalties such as in
 Jeff Fessler's papers. It should have identical dimensions to the image for which the
 penalty is computed. If \f$\kappa\f$ is not set, this class will effectively
 use 1 for all \f$\kappa\f$'s.

 By default, a 3x3 or 3x3x3 neighbourhood is used where the weights are set to
 x-voxel_size divided by the Euclidean distance between the points.

 \par Parsing
 These are the keywords that can be used in addition to the ones in GeneralPrior.
 \verbatim
 Logcosh Prior Parameters:=
 ; next defaults to 0, set to 1 for 2D inverse Euclidean weights, 0 for 3D
 only 2D:= 0
 ; scalar controls the transition between the quadratic (smooth) and linear (edge-preserving) nature of the function
 ; scalar:=
 ; next can be used to set weights explicitly. Needs to be a 3D array (of floats).
 ' value of only_2D is ignored
 ; following example uses 2D 'nearest neighbour' penalty
 ; weights:={{{0,1,0},{1,0,1},{0,1,0}}}
 ; use next parameter to specify an image with penalisation factors (a la Fessler)
 ; see class documentation for more info
 ; kappa filename:=
 ; use next parameter to get gradient images at every subiteration
 ; see class documentation
 gradient filename prefix:=
 END Logcosh Prior Parameters:=
 \endverbatim


 */
template <typename elemT>
class LogcoshPrior:  public
                     RegisteredParsingObject<
                        LogcoshPrior<elemT>,
                        GeneralisedPrior<DiscretisedDensity<3,elemT> >,
                        PriorWithParabolicSurrogate<DiscretisedDensity<3,elemT> >
                     >
{
private:
    typedef
    RegisteredParsingObject< LogcoshPrior<elemT>,
            GeneralisedPrior<DiscretisedDensity<3,elemT> >,
            PriorWithParabolicSurrogate<DiscretisedDensity<3,elemT> > >
            base_type;

public:
    //! Name which will be used when parsing a GeneralisedPrior object
    static const char * const registered_name;

    //! Default constructor
    LogcoshPrior();

    //! Constructs it explicitly
    LogcoshPrior(const bool only_2D, float penalization_factor);

    //! Constructs it explicitly with scalar
    LogcoshPrior(const bool only_2D, float penalization_factor, const float scalar);

    //! compute the value of the function
    double
    compute_value(const DiscretisedDensity<3,elemT> &current_image_estimate);

    //! compute gradient
    void compute_gradient(DiscretisedDensity<3,elemT>& prior_gradient,
                          const DiscretisedDensity<3,elemT> &current_image_estimate);

    //! compute the parabolic surrogate for the prior
    void parabolic_surrogate_curvature(DiscretisedDensity<3,elemT>& parabolic_surrogate_curvature,
                                       const DiscretisedDensity<3,elemT> &current_image_estimate);

    //! compute Hessian
    void compute_Hessian(DiscretisedDensity<3,elemT>& prior_Hessian_for_single_densel,
                         const BasicCoordinate<3,int>& coords,
                         const DiscretisedDensity<3,elemT> &current_image_estimate);

    //! Compute the multiplication of the hessian of the prior multiplied by the input.
    virtual Succeeded accumulate_Hessian_times_input(DiscretisedDensity<3,elemT>& output,
                                                     const DiscretisedDensity<3,elemT>& current_estimate,
                                                     const DiscretisedDensity<3,elemT>& input) const;

    //! get penalty weights for the neigbourhood
    Array<3,float> get_weights() const;

    //! set penalty weights for the neigbourhood
    void set_weights(const Array<3,float>&);

    //! get current kappa image
    /*! \warning As this function returns a shared_ptr, this is dangerous. You should not
     modify the image by manipulating the image referred to by this pointer.
     Unpredictable results will occur.
     */
    shared_ptr<DiscretisedDensity<3,elemT> > get_kappa_sptr() const;

    //! set kappa image
    void set_kappa_sptr(const shared_ptr<DiscretisedDensity<3,elemT> >&);

    //! Get the scalar value
    float get_scalar() const;

    //! Set the scalar value
    void set_scalar(float scalar_v);

protected:
    //! can be set during parsing to restrict the weights to the 2D case
    bool only_2D;

    //! controls the transition between the quadratic (smooth) and linear (edge-preserving) nature of the prior
    float scalar;

    //! filename prefix for outputting the gradient whenever compute_gradient() is called.
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

    virtual void set_defaults();
    virtual void initialise_keymap();
    virtual bool post_processing();

private:
    //! Spatially variant penalty penalty image ptr
    shared_ptr<DiscretisedDensity<3,elemT> > kappa_ptr;

    //! The Log(cosh()) function and its approximation
    /*!
     Cosh(x) = 0.5(e^x + e^-x) is an exponential function and hence cannot be evaluated for large x.
     Make the approximation:
     log(Cosh(x)) = log(0.5) + |x| + log(1 + e^(-2|x|)) = log(0.5) + |x| + O(10^(-27)), for |x|>30
    */
    static inline float logcosh(const float d)
    {
      const float x = fabs(d);
      if ( x < 30.f ){
        return log(cosh(x));
      } else {
        return x + log(0.5f);
      }
    }

    //! The surrogate of the logcosh function is tanh(x)/x
    /*!
     * @param d should be the difference between the ith and jth voxel.
     However, it will use the taylor expansion if the x is too small (to prevent division by 0).
     * @param scalar is the logcosh scalar value controlling the priors transition between the quadratic and linear behaviour
     * @return the surrogate of the log-cosh function
    */
    static inline float surrogate(const float d, const float scalar)
    {
      const float eps = 0.01;
      const float x = d * scalar;
      // If abs(x) is less than eps,
      // use Taylor approximatation of tanh: tanh(x)/x ~= (x - x^3/3)/x = 1- x^2/3.
      // Prevents divide by zeros
      if (fabs(x)<eps)
      { return 1- square(x)/3; }
      else
      { return tanh(x)/x; }
    }

    //! The Hessian of log(cosh()) is sech^2(x) = (1/cosh(x))^2
    /*!
     This function returns the hessian of the logcosh function
     * @param d the difference between the ith and jth voxel.
     * @param scalar is the logcosh scalar value controlling the priors transition between the quadratic and linear behaviour
     * @return the second derivative of the log-cosh function
     */
    static inline float Hessian(const float d, const float scalar)
    {
      const float x = d * scalar;
      return square((1/ cosh(x)));
    }
};


END_NAMESPACE_STIR

#endif