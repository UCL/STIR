//
//
/*
    Copyright (C) 2003- 2007, Hammersmith Imanet Ltd
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
  \ingroup normalisation

  \brief Implementation for class stir::BinNormalisation

  \author Kris Thielemans
*/


#include "stir/recon_buildblock/BinNormalisation.h"
#include "stir/RelatedViewgrams.h"
#include "stir/Bin.h"
#include "stir/Succeeded.h"

START_NAMESPACE_STIR

BinNormalisation::
~BinNormalisation()
{}

Succeeded
BinNormalisation::
set_up(const shared_ptr<ProjDataInfo>& )
{
  return Succeeded::yes;  
}


void 
BinNormalisation::apply(RelatedViewgrams<float>& viewgrams,
			const double start_time, const double end_time) const 
{
  for (RelatedViewgrams<float>::iterator iter = viewgrams.begin(); iter != viewgrams.end(); ++iter)
  {
    Bin bin(iter->get_segment_num(),iter->get_view_num(), 0,0);
    for (bin.axial_pos_num()= iter->get_min_axial_pos_num(); 
	 bin.axial_pos_num()<=iter->get_max_axial_pos_num(); 
	 ++bin.axial_pos_num())
      for (bin.tangential_pos_num()= iter->get_min_tangential_pos_num(); 
	   bin.tangential_pos_num()<=iter->get_max_tangential_pos_num(); 
	   ++bin.tangential_pos_num())
        (*iter)[bin.axial_pos_num()][bin.tangential_pos_num()] /= 
          std::max(1.E-20F, get_bin_efficiency(bin, start_time, end_time));
  }
}

void 
BinNormalisation::
undo(RelatedViewgrams<float>& viewgrams,const double start_time, const double end_time) const 
{
  for (RelatedViewgrams<float>::iterator iter = viewgrams.begin(); iter != viewgrams.end(); ++iter)
  {
    Bin bin(iter->get_segment_num(),iter->get_view_num(), 0,0);
    for (bin.axial_pos_num()= iter->get_min_axial_pos_num(); 
	 bin.axial_pos_num()<=iter->get_max_axial_pos_num(); 
	 ++bin.axial_pos_num())
      for (bin.tangential_pos_num()= iter->get_min_tangential_pos_num(); 
	   bin.tangential_pos_num()<=iter->get_max_tangential_pos_num(); 
	   ++bin.tangential_pos_num())
         (*iter)[bin.axial_pos_num()][bin.tangential_pos_num()] *= 
	   this->get_bin_efficiency(bin,start_time, end_time);
  }

}
 
END_NAMESPACE_STIR

