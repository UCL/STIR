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
START_NAMESPACE_STIR

Radionuclide::Radionuclide()
{}


Radionuclide::Radionuclide(const std::string &rname, float renergy, float rbranching_ratio, float rhalf_life,
                           ImagingModality rmodality)
	 :name(rname),energy(renergy),
      branching_ratio(rbranching_ratio),
      half_life(rhalf_life),modality(rmodality)
     {}

     
std::string
 Radionuclide:: get_name()const
{ return name;}

float
 Radionuclide::get_energy()const 
{return energy;}

float
 Radionuclide::get_branching_ratio()  const
{ return branching_ratio;}

float
 Radionuclide::get_half_life() const
{ return half_life;}

ImagingModality
 Radionuclide::get_modality() const
{ return modality;}

bool  
Radionuclide::operator==(const Radionuclide& r) const{ 
    return name==r.name && abs(energy-r.energy)<= 10e-5 && abs(branching_ratio-r.branching_ratio)<= 10e-5 &&
           abs(half_life-r.half_life)<= 10e-5 && modality==r.modality;
}

END_NAMESPACE_STIR
