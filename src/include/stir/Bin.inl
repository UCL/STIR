//
//
/*!

  \file
  \ingroup projdata

  \brief Implementations of inline functions of class stir::Bin

  \author Sanida Mustafovic
  \author Kris Thielemans
  \author PARAPET project

*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2009, Hammersmith Imanet Ltd
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

Bin::Bin()
{}


Bin::Bin(int segment_num,int view_num, int axial_pos_num,int tangential_pos_num,float bin_value)
	 :segment(segment_num),view(view_num),
	 axial_pos(axial_pos_num),tangential_pos(tangential_pos_num),bin_value(bin_value)
     {}

     
int
 Bin:: axial_pos_num()const
{ return axial_pos;}

int
 Bin::segment_num()const 
{return segment;}

int
 Bin::tangential_pos_num()  const
{ return tangential_pos;}

int
 Bin::view_num() const
{ return view;}

int&
 Bin::axial_pos_num()
{ return axial_pos;}

int&
 Bin:: segment_num()
{return segment;}

int&
 Bin::tangential_pos_num()
{ return tangential_pos;}

int&
 Bin:: view_num() 
{ return view;}

#if 0
const ProjDataInfo *
Bin::get_proj_data_info_ptr() const
{
  return proj_data_info_ptr.get();
}
#endif

Bin
Bin::get_empty_copy() const
{
 
  Bin copy(segment_num(),view_num(),axial_pos_num(),tangential_pos_num(),0);

  return copy;
}

float 
Bin::get_bin_value()const 
{ return bin_value;}

void
Bin::set_bin_value( float v )
{ bin_value = v ;}

Bin&  
Bin::operator+=(const float dx) 
{ bin_value+=dx;  
  return *this;}


bool  
Bin::operator==(const Bin& bin2) const
{ 
  return 
    segment == bin2.segment && view == bin2.view && 
    axial_pos == bin2.axial_pos && tangential_pos == bin2.tangential_pos &&
    bin_value == bin2.bin_value;
}

bool  
Bin::operator!=(const Bin& bin2) const
{ 
  return !(*this==bin2);
}

Bin&
Bin::operator*=(const float dx)
{
    bin_value*=dx;
    return *this;
}

Bin&
Bin::operator/=(const float dx)
{
    if (dx == 0.f)
        bin_value = 0.0f;
    else
        bin_value /= dx;

    return *this;
}

END_NAMESPACE_STIR
