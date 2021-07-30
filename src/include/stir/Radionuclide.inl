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
   /* Note: note checking on `name` in case it is not standardised */
//    if(!(abs(energy - r.energy)<= 1E-1)){
//        std::cout<<energy<<"   "<<r.energy<<std::endl;
//        error("energy is different" );}
//    if(!(abs(branching_ratio-r.branching_ratio)<= 1E-1))
//        error("BR is different" );
//    if(!(modality.get_modality()==r.modality.get_modality()))
//        error("modality is different" );
//    return true;
        return abs(energy - r.energy)<= 1E-1 &&
           abs(branching_ratio-r.branching_ratio)<= 1E-1 &&
           abs(half_life-r.half_life)<=1 &&
           modality.get_modality()==r.modality.get_modality();
}

END_NAMESPACE_STIR
