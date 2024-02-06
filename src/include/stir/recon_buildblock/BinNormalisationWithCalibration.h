//
//
/*
    Copyright (C) 2020-2021, University College London
    Copyright (C) 2020, National Physical Laboratory
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

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
#include "stir/Radionuclide.h"
#include "stir/shared_ptr.h"
#include "stir/recon_buildblock/BinNormalisation.h"
#include "stir/decay_correction_factor.h"

START_NAMESPACE_STIR

/*!
  \ingroup normalisation
This class provides the facility to use a calibration factor and (isotope) branching ratio when normalising data.
  Therefore, if they are set correctly, the reconstructed image will be calibrated as well.

  Note that it is the responsibility of the derived classes to set the calibration factor.
  The branching ratio is obtained from the radionuclide set in \c ExamInfo (passed by set_up()).
  */
class BinNormalisationWithCalibration : public BinNormalisation
{
private:
  using base_type = BinNormalisation;

public:
  BinNormalisationWithCalibration();
  //! initialises the object and checks if it can handle such projection data
  /*! Computes internal numbers related to calibration etc. */
  Succeeded set_up(const shared_ptr<const ExamInfo>& exam_info_sptr, const shared_ptr<const ProjDataInfo>&) override;
  //! product of calibration factor etc
  float get_calib_decay_branching_ratio_factor(const Bin&) const; // TODO find a better name
  float get_calibration_factor() const override;
  float get_branching_ratio() const;

  void set_calibration_factor(const float);
  void set_radionuclide(const Radionuclide&);

  // needs to be implemented by derived class
  virtual float get_uncalibrated_bin_efficiency(const Bin&) const = 0;

  //! return efficiency for 1 bin
  /*! returns get_uncalibrated_bin_efficiency(bin)/get_calib_decay_branching_ratio_factor(bin)
   */
  float get_bin_efficiency(const Bin& bin) const final
  {
    return this->get_uncalibrated_bin_efficiency(bin) / this->_calib_decay_branching_ratio;
  }

protected:
  // parsing stuff
  void set_defaults() override;
  void initialise_keymap() override;
  bool post_processing() override;

private:
  // provide facility to switch off things?
  //  need to be added to the parsing keywords
  //  bool use_calibration_factor; // default to true
  float calibration_factor;
  Radionuclide radionuclide;
  //! product of various factors
  /*! computed by set_up() */
  float _calib_decay_branching_ratio;
};

END_NAMESPACE_STIR

#endif
