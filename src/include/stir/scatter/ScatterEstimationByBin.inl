////
////
///*
//    Copyright (C) 2004 - 2012, Hammersmith Imanet Ltd
//    This file is part of STIR.

//    This file is free software; you can redistribute it and/or modify
//    it under the terms of the GNU Lesser General Public License as published by
//    the Free Software Foundation; either version 2.1 of the License, or
//    (at your option) any later version.

//    This file is distributed in the hope that it will be useful,
//    but WITHOUT ANY WARRANTY; without even the implied warranty of
//    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//    GNU Lesser General Public License for more details.

//    See STIR/LICENSE.txt for details
//*/
///*!
//  \file
//  \ingroup scatter
//  \brief Inline implementations of class stir::ScatterEstimationByBin (cross sections)
  
//  \author Nikolaos Dikaios
//  \author Charalampos Tsoumpas
//  \author Kris Thielemans

//*/
//START_NAMESPACE_STIR


//float
//ScatterEstimationByBin::
//dif_Compton_cross_section(const float cos_theta, float energy)
//{
//  const double Re = 2.818E-13;   // aktina peristrofis electroniou gia to atomo tou H
//  const double sin_theta_2= 1-cos_theta*cos_theta ;
//  const double P= 1/(1+(energy/511.0)*(1-cos_theta));
//  return static_cast<float>( (Re*Re/2) * P * (1 - P * sin_theta_2 + P * P));
//}

//float
//ScatterEstimationByBin::
//photon_energy_after_Compton_scatter(const float cos_theta, const float energy)
//{
//  return static_cast<float>(energy/(1+(energy/511.0)*(1-cos_theta)));   // For an arbitrary energy
//}

//float
//ScatterEstimationByBin::
//photon_energy_after_Compton_scatter_511keV(const float cos_theta)
//{
//  return 511/(2-cos_theta); // for a given energy, energy := 511 keV
//}

//float
//ScatterEstimationByBin::
//total_Compton_cross_section(const float energy)
//{
//  const double a= energy/511.0;
//  const double l= log(1.0+2.0*a);
//  const double sigma0= 6.65E-25;   // sigma0=8*pi*a*a/(3*m*m)
//  return static_cast<float>( 0.75*sigma0  * ( (1.0+a)/(a*a)*( 2.0*(1.0+a)/(1.0+2.0*a)- l/a ) + l/(2.0*a) - (1.0+3.0*a)/(1.0+2.0*a)/(1.0+2.0*a) ) ); // Klein - Nishina formula = sigma / sigma0
//}


//float
//ScatterEstimationByBin::
//total_Compton_cross_section_relative_to_511keV(const float energy)
//{
//  const double a= energy/511.0;
//  static const double prefactor = 9/(-40 + 27*log(3.)); //Klein-Nishina formula for a=1 & devided with 0.75 == (40 - 27*log(3)) / 9

//  return //checked this in Mathematica
//    static_cast<float>
//	(prefactor*
//    (((-4 - a*(16 + a*(18 + 2*a)))/square(1 + 2*a) +
//      ((2 + (2 - a)*a)*log(1 + 2*a))/a)/square(a)
//     ));
//}



//END_NAMESPACE_STIR
