#ifndef __stir_analytic_SRT2D_SRT2DReconstruction_H__
#define __stir_analytic_SRT2D_SRT2DReconstruction_H__
/*
    Copyright (C) 2024, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup analytic

  \brief declares the stir::SRT2DReconstruction class

  \author Dimitra Kyriakopoulou
  \author Kris Thielemans
*/

#include "stir/recon_buildblock/AnalyticReconstruction.h"
#include "stir/RegisteredParsingObject.h"
#include <string>
#include <vector>
#include "stir/shared_ptr.h"

START_NAMESPACE_STIR

template <int num_dimensions, typename elemT>
class DiscretisedDensity;
class Succeeded;
class ProjData;

/*! \ingroup SRT2D
 \brief Reconstruction class for 2D Spline Reconstruction Technique

  The reference for the implemented PET algorithm is: Fokas, A. S., A. Iserles, and V. Marinakis. "Reconstruction algorithm for
single photon emission computed tomography and its numerical implementation." *Journal of the Royal Society Interface* 3.6 (2006):
45-54.

  STIR implementations: Initial version June 2012, 1st updated version (4-point symmetry included) November 2012, 2nd updated
version (8-point symmetry included) July 2013, 3rd updated version 2014-2016, 4th updated version 2023-2024

 \par Parameters
  \verbatim
SRT2Dparameters :=

input file := input.hs
output filename prefix := output

; output image parameters
; zoom defaults to 1
zoom := -1
; image size defaults to whole FOV
xy output image size (in pixels) := -1

; can be used to call SSRB first
; default means: call SSRB only if no axial compression is already present
;num segments to combine with ssrb := -1

END :=
  \endverbatim
*/

class SRT2DReconstruction
    : public RegisteredParsingObject<SRT2DReconstruction, Reconstruction<DiscretisedDensity<3, float>>, AnalyticReconstruction>
{
  // typedef AnalyticReconstruction base_type;
  typedef RegisteredParsingObject<SRT2DReconstruction, Reconstruction<DiscretisedDensity<3, float>>, AnalyticReconstruction>
      base_type;
#ifdef SWIG
  // work-around swig problem. It gets confused when using a private (or protected)
  // typedef in a definition of a public typedef/member
 public:
#else
private:
#endif
  typedef DiscretisedDensity<3, float> TargetT;

public:
  //! Name which will be used when parsing a SRT2DReconstruction object
  static const char* const registered_name;

  //! Default constructor (calls set_defaults())
  SRT2DReconstruction();
  /*!
    \brief Constructor, initialises everything from parameter file, or (when
    parameter_filename == "") by calling ask_parameters().
  */
  explicit SRT2DReconstruction(const std::string& parameter_filename);

  SRT2DReconstruction(const shared_ptr<ProjData>& proj_data_ptr_v, const int num_segments_to_combine = -1);
  virtual std::string method_info() const;

  virtual void ask_parameters();

  virtual Succeeded set_up(shared_ptr<TargetT> const& target_data_sptr);

protected: // make parameters protected such that doc shows always up in doxygen
  // parameters used for parsing

  //! number of segments to combine (with SSRB) before starting 2D reconstruction
  /*! if -1, a value is chosen depending on the axial compression.
      If there is no axial compression, num_segments_to_combine is
      effectively set to 3, otherwise it is set to 1.
      \see SSRB
  */
  int num_segments_to_combine;

private:
  Succeeded actual_reconstruct(shared_ptr<DiscretisedDensity<3, float>> const& target_image_ptr);

  virtual void set_defaults();
  virtual void initialise_keymap();
  virtual bool post_processing();

  /*!
  \brief Computes second derivatives for natural cubic spline interpolation.

  This function precomputes the second derivatives of the input data points
  for use in cubic spline interpolation. The results are stored in the \a y2 vector.

  \param x Vector of x-coordinates of the input data points.
  \param y Vector of y-coordinates of the input data points.
  \param n The number of data points.
  \param y2 Vector to store the computed second derivatives for spline interpolation.
*/
  void spline(const std::vector<float>& x, const std::vector<float>& y, int n, std::vector<float>& y2) const;

  /*!
  \brief Computes the Hilbert transform derivative for a set of projections.

  This function calculates the derivative of the Hilbert transform for a set of sampled data points.
  It uses second derivatives, logarithmic differences, and a correction term to adjust the
  computed derivative.

  \param x The x-coordinate for which the derivative is evaluated.
  \param f Vector of function values at sampling points.
  \param ddf Vector of second derivatives of the function.
  \param p Vector of sampling positions.
  \param sp The total number of sampling points.
  \param lg Vector of logarithmic differences used for interpolation.
  \param termC Correction term used for adjusting the derivative.
  \return The computed Hilbert transform derivative at \a x.
*/
  float hilbert_der(float x,
                    const std::vector<float>& f,
                    const std::vector<float>& ddf,
                    const std::vector<float>& p,
                    int sp,
                    const std::vector<float>& lg,
                    float termC) const;

  /*!
  \brief Performs numerical integration over a set of sampled data.

  This function uses a simple summation approach to numerically calculate integrals over tangential positions.

  \param dist The interval over which the integration is performed.
  \param max The number of data points to integrate.
  \param ff Vector containing the function values to be integrated.
  \return The computed integral value.
*/
  float integ(float dist, int max, const std::vector<float>& ff) const;
};

END_NAMESPACE_STIR

#endif
