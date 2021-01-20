//
//

/*!
  \file

  \brief Declaration of class stir::Radionuclide
  

  \author Daniel Deidda
  \author Kris Thielemans
*/
/*
    Copyright (C) 2020 National Physical Laboratory
    Copyright (C) 2020 University College London
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
#ifndef __stir_Radionuclide_H__
#define __stir_Radionuclide_H__


#include "stir/common.h"
#include "stir/ImagingModality.h"


START_NAMESPACE_STIR
/*!   \ingroup projdata
 \brief
 A class for storing radionuclide information.

*/

class Radionuclide
{
public: 
  //! default constructor
  inline Radionuclide();

  //!  A constructor : constructs a radionuclide with all itss information
  inline Radionuclide(std::string name, float energy, float branching_ratio, float half_life,
    ImagingModality modality);
  
  //!get name
  inline std::string get_name()const;
  //! get energy
  inline float get_energy()const; 
  //! get branching_ratio
  inline float get_branching_ratio()  const; 
  //! get half_life
  inline float get_half_life() const; 
  //! get modality
  inline ImagingModality get_modality() const; 

    
  //! comparison operators
  inline bool operator==(const Radionuclide& r) const;
  
private :
  
  std::string name;
  float energy;
  float branching_ratio;
  float half_life;
  ImagingModality modality;
  
  
};



END_NAMESPACE_STIR


#include "stir/Radionuclide.inl"

#endif //__Radionuclide_H__
