//
//
#ifndef __stir_analytic_DFM3D_DFM3DReconstruction_H__
#define __stir_analytic_DFM3D_DFM3DReconstruction_H__
/*
    Copyright (C) 2024, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup analytic

  \brief declares the stir::DFM3DReconstruction class

  \author Dimitra Kyriakopoulou

*/

#include "stir/recon_buildblock/AnalyticReconstruction.h"
#include "stir/recon_buildblock/BackProjectorByBin.h"
#include "stir/RegisteredParsingObject.h"
#include <string>
#include "stir/shared_ptr.h"
#include "stir/Array_complex_numbers.h"


START_NAMESPACE_STIR

template <int num_dimensions, typename elemT> class DiscretisedDensity;
class Succeeded;
class ProjData;

class DFM3DReconstruction : 
	public 
		RegisteredParsingObject<
			DFM3DReconstruction, 
				Reconstruction < DiscretisedDensity < 3,float> >,
				AnalyticReconstruction
			>
{
  //typedef AnalyticReconstruction base_type;
	typedef
    RegisteredParsingObject<
        DFM3DReconstruction,
            Reconstruction < DiscretisedDensity < 3,float> >,
            AnalyticReconstruction
         > base_type;
	typedef DiscretisedDensity < 3,float> TargetT;
public:
	//! Name which will be used when parsing a ProjectorByBinPair object
    static const char * const registered_name;
	
  //! Default constructor (calls set_defaults())
  DFM3DReconstruction (); 
  /*!
    \brief Constructor, initialises everything from parameter file, or (when
    parameter_filename == "") by calling ask_parameters().
  */
  explicit 
    DFM3DReconstruction(const std::string& parameter_filename);

  DFM3DReconstruction(const shared_ptr<ProjData>&, 
		      const int noise_filter=0,
		      const int num_segments_to_combine=-1
		      );
  
  virtual std::string method_info() const;

  virtual void ask_parameters();  

  virtual Succeeded set_up(shared_ptr <TargetT > const& target_data_sptr);

 protected: // make parameters protected such that doc shows always up in doxygen
  // parameters used for parsing

  // noise filter 
  int noise_filter;
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
  bool post_processing_only_DFM3D_parameters();

void process_chunk(int start, int end, int sp, int sphi, int sth, int sa);
  template <typename T> void fftshift(Array< 1 , T >& a, int size);
  template <typename T> void fftshift(Array< 2 , std::complex< T > >& a, int M, int N);
  template <typename T> void fftshift(Array< 3 , std::complex< T > >& a, int M, int N, int K);
Array< 2, std::complex<float> > IDWa(const std::vector<float>& xt1, const std::vector<float>& yt1, const std::vector<std::complex<float>>& ft1, int N, const std::vector<float>& xn, const std::vector<float>& yn, int L, float p, int sp);
};




END_NAMESPACE_STIR

    
#endif
