//
//
/*!

  \file
  \brief Implementations of inline functions of class stir::Radionuclide

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
#include <sstream>
#include "stir/Radionuclide.h"

START_NAMESPACE_STIR

std::string
Radionuclide::parameter_info() const
{
  std::ostringstream s;
  s << "Modality: " << this->modality.get_name() << '\n';
  s << "Radionuclide: " << this->name << '\n';
  s << "Energy " << std::fixed << std::setprecision(12) << this->energy << std::setprecision(5) << '\n';
  s << "Half-life: " << std::fixed << std::setprecision(12) << this->half_life << std::setprecision(5) << '\n';
  s << "Branching ratio: " << std::fixed << std::setprecision(12) <<this->branching_ratio << std::setprecision(5)<<'\n';
  return s.str();
}

Radionuclide::Radionuclide()
    :name("Unknown"),energy(-1),
     branching_ratio(-1),
     half_life(-1),modality(ImagingModality::Unknown)
{}


Radionuclide::Radionuclide(const std::string &rname, float renergy, float rbranching_ratio, float rhalf_life,
                           ImagingModality rmodality)
	 :name(rname),energy(renergy),
      branching_ratio(rbranching_ratio),
      half_life(rhalf_life),modality(rmodality)
     {}

     
std::string
 Radionuclide:: get_name()const
{ 
//    if (name=="Unknown")
//        error("Radionuclide is Unknown, If you want to use it, it needs to be defined!");
    return name;
}

float
 Radionuclide::get_energy()const 
{
    if (energy<=0)
        error("Radionuclide energy peak is unset, If you want to use it, it needs to be set!");
    return energy;
}

float
 Radionuclide::get_branching_ratio()  const
{ 
    if (energy<=0)
        error("Radionuclide Branching ratio is unset, If you want to use it, it needs to be set!");
    return branching_ratio;
}

float
 Radionuclide::get_half_life() const
{
    if (half_life<=0)
        error("Radionuclide half life is unset, If you want to use it, it needs to be set!");
    return half_life;
}

ImagingModality
 Radionuclide::get_modality() const
{ 
    if (modality.is_unknown())
        error("Radionuclide::modality is Unknown, If you want to use it, it needs to be defined!");
    return modality;
}

bool  
Radionuclide::operator==(const Radionuclide& r) const{ 
        return  (abs(energy - r.energy)<= 1E-1 || (energy<=0 && r.energy <=0)) &&
           (abs(branching_ratio-r.branching_ratio)<= 1E-1 || (branching_ratio<=0 && r.branching_ratio<=0)) &&
           (abs(half_life-r.half_life)<=1 || (r.half_life<=0 && r.half_life<=0)) &&
           (modality==r.modality);
}

END_NAMESPACE_STIR