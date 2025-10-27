//
//
#ifndef __stir_analytic_GRD2D_GRD2DReconstruction_H__
#define __stir_analytic_GRD2D_GRD2DReconstruction_H__
/*
    Copyright (C) 2025, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0
 
    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup GRD2D

  \brief declares the stir::GRD2DReconstruction class

  \details
  GRD2D maps each PET view into Fourier space, interpolates non-uniform samples to a Cartesian
  grid using a Kaiser–Bessel kernel, then applies an inverse 2D FFT and resamples to the output image. A radial
  low-pass noise filter can suppress high-frequency noise. 

  The algorithm, its reference, and comments on its implementation are described in Chapter 5 of Dimitra Kyriakopoulou's doctoral thesis, “Analytical and Numerical Aspects of Tomography”, University College London (UCL), 2024, supervised by Professor Athanassios S. Fokas (Cambridge) and Professor Kris Thielemans (UCL). Available at: https://discovery.ucl.ac.uk/id/eprint/10202525/
  
  \author Dimitra Kyriakopoulou

*/

#include "stir/recon_buildblock/AnalyticReconstruction.h"
//#include "stir/recon_buildblock/BackProjectorByBin.h"
#include "stir/RegisteredParsingObject.h"
#include <string>
#include "stir/shared_ptr.h"
//#include "stir/Array_complex_numbers.h"
//#include "stir/numerics/fftshift.h"


START_NAMESPACE_STIR

template <int num_dimensions, typename elemT> class DiscretisedDensity;
class Succeeded;
class ProjData;

class GRD2DReconstruction : 
	public 
		RegisteredParsingObject<
			GRD2DReconstruction, 
				Reconstruction < DiscretisedDensity < 3,float> >,
				AnalyticReconstruction
			>
{
  //typedef AnalyticReconstruction base_type;
	typedef
    RegisteredParsingObject<
        GRD2DReconstruction,
            Reconstruction < DiscretisedDensity < 3,float> >,
            AnalyticReconstruction
         > base_type;
	
public:
	//! Name which will be used when parsing a ProjectorByBinPair object
    static const char * const registered_name;
	
  //! Default constructor (calls set_defaults())
  GRD2DReconstruction (); 
  /*!
    \brief Constructor, initialises everything from parameter file, or (when
    parameter_filename == "") by calling ask_parameters().
  */
  explicit 
    GRD2DReconstruction(const std::string& parameter_filename);

  GRD2DReconstruction(const shared_ptr<ProjData>&, 
		      const double noise_filter=-1.,
			  const double alpha_gridding=1.,
			  const double kappa_gridding=4.,
		      const int num_segments_to_combine=-1
		      );
  
  virtual std::string method_info() const;

  virtual void ask_parameters();

  virtual Succeeded set_up(shared_ptr <TargetT > const& target_data_sptr);

 protected: // make parameters protected such that doc shows always up in doxygen
  // parameters used for parsing

  // alpha and kappa for gridding
  double alpha_gridding; 
  double kappa_gridding; 
  // noise filter 
  double noise_filter;
	
  //! number of segments to combine (with SSRB) before starting 2D reconstruction
  /*! if -1, a value is chosen depending on the axial compression.
      If there is no axial compression, num_segments_to_combine is
      effectively set to 3, otherwise it is set to 1.
      \see SSRB
  */
  int num_segments_to_combine;
  //! potentially display data
  /*! allowed values: \c display_level=0 (no display), 1 (only final image), 
      2 (filtered-viewgrams). Defaults to 0.
   */
  int display_level;
 private:
  Succeeded actual_reconstruct(shared_ptr<DiscretisedDensity<3,float> > const & target_image_ptr);


  virtual void set_defaults();
  virtual void initialise_keymap();
  virtual bool post_processing(); 
  bool post_processing_only_GRD2D_parameters();

};




END_NAMESPACE_STIR

    
#endif
