/*
    Copyright (C) 2013, University College London
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
  Class for encoding the modality
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
  explicit ImagingModality(const std::string& modality_string_v)
    : modality_string(modality_string_v)
    {
      this->set_enum();
    }

    ImagingModalityValue get_modality() const
    {
      return this->modality;
    }

    std::string get_name() const
      {
        return this->modality_string;
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

  void set_enum()
  {
    const std::string mod = standardise_interfile_keyword(this->modality_string);
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
      this->modality=Unknown;
  }
};

} // namespace
#endif
