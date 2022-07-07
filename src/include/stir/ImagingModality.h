/*
    Copyright (C) 2013, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup buildblock

  \brief Declaration of class stir::ImagingModality

  \author Kris Thielemans
*/

#ifndef __stir_ImagingModality_H__
#define __stir_ImagingModality_H__

#include "stir/interfile_keyword_functions.h"
#include <string>

namespace stir {

/*! \ingroup buildblock
  \brief Class for encoding the modality.

  Modality-names follow DICOM conventions, i.e.
  "PT", "NM", "MR", "CT", "US", "Optical".
*/
class ImagingModality
{
 public:
  //! enum with possible values (using DICOM naming)
  enum ImagingModalityValue
    { Unknown, PT, NM, MR, CT, US, Optical};

  //! Construct from enum, set string accordingly
  ImagingModality(ImagingModalityValue modality_v = Unknown)
    : modality(modality_v)
    {
      this->set_string();
    }

  //! Construct from string, set enum accordingly
  /*!
    The argument is first stripped of white-space and converted to lower-case. Then
    the following are supported:
    - \c PT: "pt" or "pet"
    - \c NM: "nm" or "nucmed" or "spect"
    - \c MR: "mr" or "mri"
    - \c CT: "ct" or "cat"
    - \c US: ""us" or "ultrasound"
    - \c Optical: "optical"
    - else the modality is set to \c Unknown
  */
  explicit ImagingModality(const std::string& modality_string_v)
    {
      this->set_from_string(modality_string_v);
    }

  ImagingModalityValue get_modality() const
    {
      return this->modality;
    }

  //! Returns name as a standardised string (in DICOM conventions)
  std::string get_name() const
    {
      return this->modality_string;
    }

  bool operator==(const ImagingModality& mod) const
    {
      return this->modality == mod.modality;
    }
  bool operator!=(const ImagingModality& mod) const
    {
      return !(*this==mod);
    }
  bool is_known() const
    {
      return this->modality != Unknown;
    }
  bool is_unknown() const
    {
      return this->modality == Unknown;
    }
 private:
  ImagingModalityValue modality;
  std::string modality_string;

  void set_string()
  { 
    switch (this->modality)
      {
      case PT: this->modality_string="PT"; break;
      case NM: this->modality_string="NM"; break;
      case MR: this->modality_string="MR"; break;
      case CT: this->modality_string="CT"; break;
      case US: this->modality_string="US"; break;
      case Optical: this->modality_string="Optical"; break;
      default:
      case Unknown: this->modality_string="Unknown"; break;
      }
  }

  void set_from_string(const std::string& modality)
  {
    const std::string mod = standardise_interfile_keyword(modality);
    if (mod=="pt" || mod=="pet")
      this->modality=PT;
    else if ( mod=="nm" || mod=="nucmed" || mod=="spect")
      this->modality=NM;
    else if ( mod=="mr" || mod=="mri") 
      this->modality=MR;
    else if ( mod=="ct" || mod=="cat") 
      this->modality=CT;
    else if ( mod=="us" || mod=="ultrasound")
      this->modality=US;
    else if ( mod=="optical")
      this->modality=Optical;
    else
      {
        if (!mod.empty() && mod != "unknown")
          warning("Unrecognised modality: '" + mod + "'. Setting to Unknown");
        this->modality=Unknown;
      }
    this->set_string();
  }
};

} // namespace
#endif
