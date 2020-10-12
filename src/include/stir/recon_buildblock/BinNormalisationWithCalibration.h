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
        public  BinNormalisation
{
private:
  using base_type = BinNormalisation;
public:
    
    
  BinNormalisationWithCalibration();

 protected:
  // parsing stuff
  virtual void set_defaults();
  virtual void initialise_keymap();
  virtual bool post_processing();

   float get_calib_decay_branching_ratio_factor(const Bin&) const; // TODO find a better name
   // needs to be implemented by derived class
   virtual float get_uncalibrated_bin_efficiency(const Bin&, const double start_time, const double end_time) const  = 0;
   float get_bin_efficiency(const Bin& bin, const double start_time, const double end_time)
    { return this->get_uncalibrated_bin_efficiency(bin, start_time, end_time)/get_calib_decay_branching_ratio_factor(bin); }

   void set_calibration_factor(const float);
   void set_radionuclide(const std::string&);
private:
  // provide facility to switch off things?
  //  need to be added to the parsing keywords
  bool use_calibration_factor; // default to true
//  bool use_branching_ratio; // default to true
  float calibration_factor;
  float radionuclide;
};

END_NAMESPACE_STIR

#endif
