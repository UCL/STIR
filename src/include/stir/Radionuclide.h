/*!
  \file
  \ingroup ancillary

  \brief Declaration of class stir::Radionuclide

  \author Daniel Deidda
  \author Kris Thielemans
*/
/*
    Copyright (C) 2021 National Physical Laboratory
    Copyright (C) 2021 University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
#ifndef __stir_Radionuclide_H__
#define __stir_Radionuclide_H__

#include "stir/ImagingModality.h"
#include "stir/info.h"


START_NAMESPACE_STIR
/*!   \ingroup ancillary
 \brief
 A class for storing radionuclide information.

 \sa RadionuclideDB
*/

class Radionuclide
{
public: 
  //! default constructor
  /*! Set name to "Unknown" and other values to invalid numbers (e.g. -1) */
  Radionuclide();

  //!  A constructor : constructs a radionuclide with all its information
  Radionuclide(const std::string& name, float energy, float branching_ratio, float half_life,
    ImagingModality modality);
  
  //! get name (normally in DICOM format)
  std::string get_name()const;
  //! get energy (in keV)
  /*! if \a check=\c true, calls error() if value not set (i.e. is negative)*/
  float get_energy(bool check=true) const;
  //! get branching_ratio
  /*! if \a check=\c true, calls error() if value not set (i.e. is negative)*/
  float get_branching_ratio(bool check=true)  const;
  //! get half_life (in seconds)
  /*! if \a check=\c true, calls error() if value not set (i.e. is negative)*/
  float get_half_life(bool check=true) const;
  //! get modality
  /*! if \a check=\c true, calls error() if value not set (i.e. is unknown)*/
  ImagingModality get_modality(bool check=true) const;
  //! Return a string with info on parameters
  /*! the returned string is not intended for parsing. */
  std::string parameter_info() const;
    
  //! comparison operators
  bool operator==(const Radionuclide& r) const;
  
private :
  
  std::string name;
  float energy;
  float branching_ratio;
  float half_life;
  ImagingModality modality;
  
  
};

END_NAMESPACE_STIR
#endif //__stir_Radionuclide_H__
