//
//

/*!
  \file
  \ingroup projdata

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
/*!   \ingroup projdata
 \brief
 A class for storing radionuclide information.

*/

class Radionuclide
{
public: 
  //! default constructor
  Radionuclide();

  //!  A constructor : constructs a radionuclide with all it's information
  Radionuclide(const std::string& name, float energy, float branching_ratio, float half_life,
    ImagingModality modality);
  
  //!get name
  std::string get_name()const;
  //! get energy
  float get_energy()const; 
  //! get branching_ratio
  float get_branching_ratio()  const; 
  //! get half_life
  float get_half_life() const; 
  //! get modality
  ImagingModality get_modality() const; 
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
