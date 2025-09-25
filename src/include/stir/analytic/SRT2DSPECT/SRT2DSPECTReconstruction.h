//
//
#ifndef __stir_analytic_SRT2DSPECT_SRT2DSPECTReconstruction_H__
#define __stir_analytic_SRT2DSPECT_SRT2DSPECTReconstruction_H__
/*
    Copyright (C) 2024, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup analytic

  \brief declares the stir::SRT2DSPECTReconstruction class

  \author Dimitra Kyriakopoulou
  \author Kris Thielemans
*/

#include "stir/analytic/SRT2DSPECT/SRT2DSPECTReconstruction.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/ProjDataInfoCylindricalArcCorr.h"
#include "stir/ArcCorrection.h"
#include "stir/SSRB.h"
#include "stir/ProjDataInMemory.h"
#include "stir/Array.h"
#include <vector>
#include "stir/Sinogram.h"
#include "stir/Viewgram.h"
#include <cmath>
#include "stir/Bin.h"
#include "stir/round.h"
#include "stir/display.h"
#include <algorithm>
#include "stir/IO/interfile.h"
#include "stir/info.h"

#include "stir/recon_buildblock/AnalyticReconstruction.h"
#include "stir/RegisteredParsingObject.h"
#include <string>
#include "stir/shared_ptr.h"

START_NAMESPACE_STIR

template <int num_dimensions, typename elemT>
class DiscretisedDensity;
class Succeeded;
class ProjData;

/*! \ingroup SRT2DSPECT
 \brief Reconstruction class for 2D Spline Reconstruction Technique

  The reference for the implemented SPECT algorithm is: Fokas, A. S., A. Iserles, and V. Marinakis. "Reconstruction algorithm for
single photon emission computed tomography and its numerical implementation." *Journal of the Royal Society Interface* 3.6 (2006):
45-54.

  STIR implementations: initial version 2014-2016, 1st updated version 2023-2024

  \par Parameters

  SRT2DSPECT takes two inputs:
  - The emission sinogram, which represents the measured attenuated data.
  - The attenuation projection sinogram, which is the Radon transform (line integrals) of the attenuation map.

  \verbatim
SRT2DSPECTparameters :=

input file := input.hs
attenuation projection filename := attenuation_projection_sinogram.hs
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

class SRT2DSPECTReconstruction : public RegisteredParsingObject<SRT2DSPECTReconstruction,
                                                                Reconstruction<DiscretisedDensity<3, float>>,
                                                                AnalyticReconstruction>
{
  // typedef AnalyticReconstruction base_type;
  typedef RegisteredParsingObject<SRT2DSPECTReconstruction, Reconstruction<DiscretisedDensity<3, float>>, AnalyticReconstruction>
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
  //! Name which will be used when parsing a ProjectorByBinPair object
  static const char* const registered_name;

  //! Default constructor (calls set_defaults())
  SRT2DSPECTReconstruction();
  /*!
    \brief Constructor, initialises everything from parameter file, or (when
    parameter_filename == "") by calling ask_parameters().
  */
  explicit SRT2DSPECTReconstruction(const std::string& parameter_filename);

  SRT2DSPECTReconstruction(const shared_ptr<ProjData>& proj_data_ptr_v, const int num_segments_to_combine = -1);

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
  std::string attenuation_projection_filename;
  float thres_restr_bound;
  std::vector<double> thres_restr_bound_vector;
  shared_ptr<ProjData> atten_data_ptr;

private:
  Succeeded actual_reconstruct(shared_ptr<DiscretisedDensity<3, float>> const& target_image_ptr);

  virtual void set_defaults();
  virtual void initialise_keymap();
  virtual bool post_processing();

  /*!
    \brief Computes the Hilbert transform at a given node.

    This function calculates the Hilbert transform at a specific position
    based on the given function values and their second derivatives.

    \param x The x-coordinate for the node.
    \param f Vector of function values at sampling points.
    \param ddf Vector of second derivatives of the function.
    \param p Vector of sampling positions.
    \param sp Number of sampling positions.
    \param fn The function value at point \a x.
    \return The computed Hilbert transform value at the specified node.
  */
  float hilbert_node(
      float x, const std::vector<float>& f, const std::vector<float>& ddf, const std::vector<float>& p, int sp, float fn) const;

  /*!
  \brief Computes the Hilbert transform for a set of sampled data.

  This function calculates the Hilbert transform for a single set of sampled data points,
  which is used to adjust for phase shifts in projections during SPECT reconstruction.
  The transform is computed using the function values, their second derivatives,
  and logarithmic differences.

  \param x The x-coordinate where the transform is evaluated.
  \param f Vector containing function values at the sampling positions.
  \param ddf Vector of second derivatives of the function values.
  \param p Vector of sampling positions.
  \param sp The total number of sampling points.
  \param lg Vector of logarithmic differences used for interpolation adjustments.
  \return The computed Hilbert transform value at the specified x-coordinate.
*/
  float hilbert(float x,
                const std::vector<float>& f,
                const std::vector<float>& ddf,
                const std::vector<float>& p,
                int sp,
                std::vector<float>& lg) const;

  /*!
    \brief Computes the Hilbert transform derivatives for two sets of sampled data.

    This function calculates the Hilbert transform derivatives for two separate sets of
    sampled data points, which are used to adjust for phase shifts in projections
    during SPECT reconstruction. The derivatives are computed using the function values,
    their second derivatives, and logarithmic differences.

    \param x The x-coordinate where the derivatives are evaluated.
    \param f Vector containing function values for the first set.
    \param ddf Vector of second derivatives for the first set.
    \param f1 Vector containing function values for the second set.
    \param ddf1 Vector of second derivatives for the second set.
    \param p Vector of sampling positions.
    \param sp Total number of sampling positions.
    \param dhp Pointer to store the computed derivative for the first set.
    \param dh1p Pointer to store the computed derivative for the second set.
    \param lg Vector of logarithmic differences used for interpolation.
  */
  void hilbert_der_double(float x,
                          const std::vector<float>& f,
                          const std::vector<float>& ddf,
                          const std::vector<float>& f1,
                          const std::vector<float>& ddf1,
                          const std::vector<float>& p,
                          int sp,
                          float* dhp,
                          float* dh1p,
                          const std::vector<float>& lg) const;

  /*!
  \brief Performs interpolation using precomputed cubic splines.

  This function uses the precomputed second derivatives (calculated by the \a spline function)
  to interpolate a value at a specified x-coordinate. It returns the interpolated value.

  \param xa The x-coordinates of the original data points.
  \param ya The y-coordinates of the original data points.
  \param y2a Precomputed second derivatives from the \a spline function.
  \param n The number of data points.
  \param x The x-coordinate where the interpolation is evaluated.
  \return The interpolated function value at \a x.

  \note This function relies on the results of the \a spline function to efficiently compute the interpolation.
*/
  float splint(const std::vector<float>& xa, const std::vector<float>& ya, const std::vector<float>& y2a, int n, float x) const;

  /*!
  \brief Computes second derivatives for natural cubic spline interpolation.

  This function precomputes the second derivatives of the input data points
  for use in cubic spline interpolation. The results are stored in the \a y2 vector.

  \param x The x-coordinates of the input data points.
  \param y The y-coordinates of the input data points.
  \param n The number of data points.
  \param y2 Vector to store the computed second derivatives for spline interpolation.

  \note This function prepares the data for efficient use in the \a splint function.
*/
  void spline(const std::vector<float>& x, const std::vector<float>& y, int n, std::vector<float>& y2) const;

  /*!
    \brief Performs numerical integration over a specified range.

    This function computes numerical integration using a summation approach.

    \param dist The distance interval over which to perform the integration.
    \param max The number of discrete data points used for the integration.
    \param ff Array containing function values at each sampled point.
    \return The computed integral over the specified range.
  */
  float integ(float dist, int max, float ff[]) const;
};

END_NAMESPACE_STIR

#endif
