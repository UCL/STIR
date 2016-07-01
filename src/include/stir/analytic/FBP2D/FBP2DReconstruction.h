//
//
#ifndef __stir_analytic_FBP2D_FBP2DReconstruction_H__
#define __stir_analytic_FBP2D_FBP2DReconstruction_H__
/*
    Copyright (C) 2000 PARAPET partners
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

    See STIR/LICENSE.txt for details
*/
/*!
  \file 
  \ingroup FBP2D
 
  \brief declares the stir::FBP2DReconstruction class

  \author Kris Thielemans
  \author PARAPET project

*/

#include "stir/recon_buildblock/AnalyticReconstruction.h"
#include "stir/recon_buildblock/BackProjectorByBin.h"
#include "stir/RegisteredParsingObject.h"
#include <string>
#include "stir/shared_ptr.h"

START_NAMESPACE_STIR

template <int num_dimensions, typename elemT> class DiscretisedDensity;
class Succeeded;
class ProjData;

/*! \ingroup FBP2D
 \brief Reconstruction class for 2D Filtered Back Projection

  \par Parameters
  \verbatim
fbp2dparameters :=

input file := input.hs
output filename prefix := output

; output image parameters
; zoom defaults to 1
zoom := 1
; image size defaults to whole FOV
xy output image size (in pixels) := 180

; can be used to call SSRB first
; default means: call SSRB only if no axial compression is already present
;num segments to combine with ssrb := -1

; filter parameters, default to pure ramp
alpha parameter for ramp filter := 1
cut-off for ramp filter (in cycles) := 0.5

; allow less padding. DO NOT USE 
; (unless you're sure that the object occupies only half the FOV)
;Transaxial extension for FFT:=1

; back projector that could be used (defaults to interpolating backprojector)
; Back projector type:= some type

; display data during processing for debugging purposes
; Display level := 0
end := 
  \endverbatim

  alpha specifies the usual Hamming window (although I'm not so sure about the terminology here). So, 
  for the "ramp filter" alpha =1. In frequency space, something like (from RampFilter.cxx)

  \code
   (alpha + (1 - alpha) * cos(_PI * f / fc))
  \endcode
 
*/
class FBP2DReconstruction :
        public
            RegisteredParsingObject<
                FBP2DReconstruction,
                    Reconstruction < DiscretisedDensity < 3,float> >,
                    AnalyticReconstruction
                 >
{
  //typedef AnalyticReconstruction base_type;
    typedef
    RegisteredParsingObject<
        FBP2DReconstruction,
            Reconstruction < DiscretisedDensity < 3,float> >,
            AnalyticReconstruction
         > base_type;
public:
    //! Name which will be used when parsing a ProjectorByBinPair object
    static const char * const registered_name;

  //! Default constructor (calls set_defaults())
  FBP2DReconstruction (); 
  /*!
    \brief Constructor, initialises everything from parameter file, or (when
    parameter_filename == "") by calling ask_parameters().
  */
  explicit 
    FBP2DReconstruction(const std::string& parameter_filename);

  FBP2DReconstruction(const shared_ptr<ProjData>&, 
		      const double alpha_ramp=1.,
		      const double fc_ramp=.5,
		      const int pad_in_s=2,
		      const int num_segments_to_combine=-1
		      );
  
  virtual std::string method_info() const;

  virtual void ask_parameters();

 protected: // make parameters protected such that doc shows always up in doxygen
  // parameters used for parsing

  //! Ramp filter: Alpha value
  double alpha_ramp;
  //! Ramp filter: Cut off frequency
  double fc_ramp;  
  //! amount of padding for the filter (has to be 0,1 or 2)
  int pad_in_s;
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

  shared_ptr<BackProjectorByBin> back_projector_sptr;

  virtual void set_defaults();
  virtual void initialise_keymap();
  virtual bool post_processing(); 
  bool post_processing_only_FBP2D_parameters();

};




END_NAMESPACE_STIR

    
#endif

