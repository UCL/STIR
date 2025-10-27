//
//
#ifndef __stir_analytic_DDSR2D_DDSR2DReconstruction_H__
#define __stir_analytic_DDSR2D_DDSR2DReconstruction_H__
/*
    Copyright (C) 2024-2025, Dimitra Kyriakopoulou
    Copyright (C) 2025, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup DDSR2D

  \brief declares the stir::DDSR2DReconstruction class

  \details
  DDSR2D reconstructs a 2D activity image from parallel-beam SPECT data with attenuation.
  It forms exponentially weighted projections, applies Hilbert transforms along the detector
  axis, differentiates with respect to the tangential coordinate, and integrates over angle
  (backprojection). Two optional frequency-domain cut-offs control smoothing.

  The algorithm, its reference, and comments on its implementation are described in Chapter 6 of Dimitra Kyriakopoulou's doctoral
  thesis, “Analytical and Numerical Aspects of Tomography”, University College London (UCL), 2024, supervised by Professor
  Athanassios S. Fokas (Cambridge) and Professor Kris Thielemans (UCL). Available at:
  https://discovery.ucl.ac.uk/id/eprint/10202525/

  \author Dimitra Kyriakopoulou

*/

#include "stir/recon_buildblock/AnalyticReconstruction.h"
//#include "stir/recon_buildblock/BackProjectorByBin.h"
#include "stir/RegisteredParsingObject.h"
#include <string>
#include "stir/shared_ptr.h"

START_NAMESPACE_STIR

template <int num_dimensions, typename elemT>
class DiscretisedDensity;
class Succeeded;
class ProjData;

class DDSR2DReconstruction
    : public RegisteredParsingObject<DDSR2DReconstruction, Reconstruction<DiscretisedDensity<3, float>>, AnalyticReconstruction>
{
  // typedef AnalyticReconstruction base_type;
  typedef RegisteredParsingObject<DDSR2DReconstruction, Reconstruction<DiscretisedDensity<3, float>>, AnalyticReconstruction>
      base_type;

public:
  static const char* const registered_name;

  //! Default constructor (calls set_defaults())
  DDSR2DReconstruction();
  /*!
    \brief Constructor, initialises everything from parameter file, or (when
    parameter_filename == "") by calling ask_parameters().
  */
  explicit DDSR2DReconstruction(const std::string& parameter_filename);

  DDSR2DReconstruction(const shared_ptr<ProjData>&,
                       const shared_ptr<DiscretisedDensity<3, float>>&,
                       const double noise_filter = -1.,
                       const double noise_filter2 = -1.);

  virtual std::string method_info() const;

  virtual void ask_parameters();

  virtual Succeeded set_up(shared_ptr<TargetT> const& target_data_sptr);

protected: // make parameters protected such that doc shows always up in doxygen
  // parameters used for parsing

  // Noise filter
  double noise_filter;
  //! Ramp filter: Alpha value
  double noise_filter2;

  //! potentially display data
  /*! allowed values: \c display_level=0 (no display), 1 (only final image),
      2 (filtered-viewgrams). Defaults to 0.
   */
  int display_level;

  std::string attenuation_map_filename;
  shared_ptr<DiscretisedDensity<3, float>> atten_data_ptr;

private:
  Succeeded actual_reconstruct(shared_ptr<DiscretisedDensity<3, float>> const& target_image_ptr);

  virtual void set_defaults();
  virtual void initialise_keymap();
  virtual bool post_processing();
  // bool post_processing_only_DDSR2D_parameters();
};

END_NAMESPACE_STIR

#endif
