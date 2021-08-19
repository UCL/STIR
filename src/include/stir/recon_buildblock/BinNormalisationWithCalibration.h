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
This class provides the facility to use a calibration factor and branching ratio when normalising data.
  Therefore, if they are set correctly, the reconstructed image will be calibrated as well.
  
  Note that it is the responsibility of the derived classes to set these factors.
  */
class BinNormalisationWithCalibration : 
        public  BinNormalisation
{
private:
  using base_type = BinNormalisation;
public:
    
    
  BinNormalisationWithCalibration();
  float get_calib_decay_branching_ratio_factor(const Bin&) const; // TODO find a better name
  float get_calibration_factor() const override;
  float get_branching_ratio() const;
  
  void set_calibration_factor(const float);
  void set_branching_ratio(const float);
  void set_radionuclide(const std::string&);
  
  // needs to be implemented by derived class
  virtual float get_uncalibrated_bin_efficiency(const Bin&) const  = 0;
 
  virtual float get_bin_efficiency(const Bin& bin) const final
   { return this->get_uncalibrated_bin_efficiency(bin)/get_calib_decay_branching_ratio_factor(bin); }
  
 protected:
  // parsing stuff
  virtual void set_defaults() override;
  virtual void initialise_keymap() override;
  virtual bool post_processing() override;


private:
  // provide facility to switch off things?
  //  need to be added to the parsing keywords
//  bool use_calibration_factor; // default to true
//  bool use_branching_ratio; // default to true
  float calibration_factor;
  float branching_ratio;
  std::string radionuclide;
};

END_NAMESPACE_STIR

#endif
