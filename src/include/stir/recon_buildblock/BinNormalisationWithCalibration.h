//
//
/*
    Copyright (C) 2020, UCL
    Copyright (C) 2020, NPL
    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup normalisation

  \brief Declaration of class stir::BinNormalisationWithCalibration

  \author Daniel Deidda
  \author Kris Thielemans
*/

#ifndef __stir_recon_buildblock_BinNormalisationWithCalibration_H__
#define __stir_recon_buildblock_BinNormalisationWithCalibration_H__


#include "stir/RegisteredParsingObject.h"
#include "stir/Bin.h"
#include "stir/shared_ptr.h"
#include "stir/recon_buildblock/BinNormalisation.h"
#include "stir/decay_correction_factor.h"

START_NAMESPACE_STIR

/*!
  \ingroup normalisation
*/
class BinNormalisationWithCalibration : 
        public RegisteredParsingObject<BinNormalisationWithCalibration, BinNormalisation >
{
private:
  using base_type = RegisteredParsingObject<BinNormalisationWithCalibration, BinNormalisation >;
public:

    //! Name which will be used when parsing a BinNormalisationWithCalibration object
    static const char * const registered_name; 
    
  BinNormalisationWithCalibration();

  //! Return the 'efficiency' factor for a single bin

  float get_bin_efficiency(const Bin& bin,const double start_time, const double end_time) const =0;

  //! normalise some data: divide by the factors 
  void apply(RelatedViewgrams<float>&,const double start_time, const double end_time) const;

  //! undo the normalisation of some data: multiply by the factors 
  void undo(RelatedViewgrams<float>&,const double start_time, const double end_time) const; 

 protected:
  // parsing stuff
  virtual void set_defaults();
  virtual void initialise_keymap();
  virtual bool post_processing();
  float calibration_factor, branching_ratio;
};

END_NAMESPACE_STIR

#endif
