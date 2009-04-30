//
// $Id$
//
/*!
  \file
  \ingroup projection
  
  \brief Inline implementations for class stir::ProjMatrixElemsForOneDenselValue
    
  \author Kris Thielemans
      
  $Date$        
  $Revision$
*/
/*
    Copyright (C) 2000- $Date$, Hammersmith Imanet Ltd
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

ProjMatrixElemsForOneDenselValue::
ProjMatrixElemsForOneDenselValue(const Bin& bin)
: Bin(bin)
{
}  

ProjMatrixElemsForOneDenselValue::
ProjMatrixElemsForOneDenselValue()
    : Bin()
{}  


  
ProjMatrixElemsForOneDenselValue& 
ProjMatrixElemsForOneDenselValue::
operator+=(const ProjMatrixElemsForOneDenselValue& el2)
{
  //TODO assert(get_coords() == el2.get_coords());
  //*this += static_cast<const Bin&>(el2);
  set_bin_value(get_bin_value() + el2.get_bin_value());
  return *this;
}

ProjMatrixElemsForOneDenselValue& 
ProjMatrixElemsForOneDenselValue::
operator+=(const float d)
{
  static_cast<Bin&>(*this) += d;
  return *this;
}

ProjMatrixElemsForOneDenselValue& 
ProjMatrixElemsForOneDenselValue::
operator*=(const float d)
{
  set_bin_value(get_bin_value() * d);
  return *this;
}

ProjMatrixElemsForOneDenselValue& 
ProjMatrixElemsForOneDenselValue::
operator/=(const float d)
{
  set_bin_value(get_bin_value() / d);
  return *this;
}

bool 
ProjMatrixElemsForOneDenselValue::
coordinates_equal(const ProjMatrixElemsForOneDenselValue& el1, const ProjMatrixElemsForOneDenselValue& el2)
{
  return 
    el1.segment_num() == el2.segment_num() &&
    el1.view_num() == el2.view_num() &&
    el1.axial_pos_num() == el2.axial_pos_num() &&
    el1.tangential_pos_num() == el2.tangential_pos_num();
}

bool 
ProjMatrixElemsForOneDenselValue::
coordinates_less(const ProjMatrixElemsForOneDenselValue& el1, const ProjMatrixElemsForOneDenselValue& el2)
{
  return 
    el1.segment_num() < el2.segment_num() ||
    (el1.segment_num() == el2.segment_num() &&
      (el1.view_num() < el2.view_num() ||
        (el1.view_num() == el2.view_num() &&
           (el1.axial_pos_num() < el2.axial_pos_num() ||
             (el1.axial_pos_num() == el2.axial_pos_num() &&
              el1.tangential_pos_num() < el2.tangential_pos_num())))));
}



bool 
operator==(const ProjMatrixElemsForOneDenselValue& el1, 
           const ProjMatrixElemsForOneDenselValue& el2)
{
  return static_cast<const Bin&>(el1) == static_cast<const Bin&>(el2);
}


bool 
operator<(const ProjMatrixElemsForOneDenselValue& el1, 
          const ProjMatrixElemsForOneDenselValue& el2) 
{
  return 
    el1.segment_num() < el2.segment_num() ||
    (el1.segment_num() == el2.segment_num() &&
      (el1.view_num() < el2.view_num() ||
        (el1.view_num() == el2.view_num() &&
           (el1.axial_pos_num() < el2.axial_pos_num() ||
             (el1.axial_pos_num() == el2.axial_pos_num() &&
              (el1.tangential_pos_num() < el2.tangential_pos_num() ||
                (el1.tangential_pos_num() < el2.tangential_pos_num() &&
                 el1.get_bin_value()<el2.get_bin_value())))))));
}

END_NAMESPACE_STIR
