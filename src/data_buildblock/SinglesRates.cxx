//
// $Id$
//
/*
    Copyright (C) 2003- $Date$, Hammersmith Imanet Ltd
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
  \ingroup singles_buildblock
  \brief Implementation of stir::FrameSinglesRates and stir::SinglesRates

  \author Kris Thielemans
  \author Tim Borgeaud
  $Date$
  $Revision$
*/

#include "stir/data/SinglesRates.h"


START_NAMESPACE_STIR


/*
 *! FrameSinglesRates constructor.
 */
FrameSinglesRates::
FrameSinglesRates(vector<float>& avg_singles_rates,
                  double start_time,
                  double end_time,
                  shared_ptr<Scanner> scanner_sptr) :
  _start_time(start_time),
  _end_time(end_time),
  _singles(avg_singles_rates),
  _scanner_sptr(scanner_sptr)
{
  assert(avg_singles_rates.size() == static_cast<std::size_t>(scanner_sptr->get_num_singles_units()));
}



float 
FrameSinglesRates::
get_singles_rate(int singles_bin_index) const {
  return(_singles[singles_bin_index]);
}

 

float 
FrameSinglesRates::
get_singles_rate(const DetectionPosition<>& det_pos) const {
  int singles_bin_index = _scanner_sptr->get_singles_bin_index(det_pos);
  return(get_singles_rate(singles_bin_index));
}




double
FrameSinglesRates::
get_start_time() const {
  return(_start_time);
}
    
double
FrameSinglesRates::
get_end_time() const {
  return(_end_time);
}



// Get the average singles rate for a particular bin.
float
SinglesRates::
get_singles_rate(const DetectionPosition<>& det_pos,
                 const double start_time, const double end_time) const 
{
  const int singles_bin_index = scanner_sptr->get_singles_bin_index(det_pos);
  return(get_singles_rate(singles_bin_index, start_time, end_time));
}



END_NAMESPACE_STIR



