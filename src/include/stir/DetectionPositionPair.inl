//
//

/*!
  \file
  \ingroup projdata
  \brief Implementation of inline methods of class stir::DetectionPositionPair
  \author Kris Thielemans
  \author Elise Emond
*/
/*
    Copyright (C) 2002- 2009, Hammersmith Imanet Ltd
    Copyright 2017, University College London
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
template <typename coordT>
DetectionPositionPair<coordT>::
DetectionPositionPair()
  : _timing_pos(static_cast<coordT>(0))
{}

template <typename coordT>
DetectionPositionPair<coordT>::
DetectionPositionPair(const DetectionPosition<coordT>& pos1,
                      const DetectionPosition<coordT>& pos2,
					  const coordT timing_pos)
  : p1(pos1), p2(pos2), _timing_pos(timing_pos)
{}

template <typename coordT>
const DetectionPosition<coordT>&
DetectionPositionPair<coordT>::
pos1() const
{ return p1; }

template <typename coordT>
const DetectionPosition<coordT>&
DetectionPositionPair<coordT>::
pos2() const
{ return p2; }

template <typename coordT>
const coordT
DetectionPositionPair<coordT>::
timing_pos() const
{ return _timing_pos; }

template <typename coordT>
DetectionPosition<coordT>&
DetectionPositionPair<coordT>::
pos1()
{ return p1; }

template <typename coordT>
DetectionPosition<coordT>&
DetectionPositionPair<coordT>::
pos2()
{ return p2; }

template <typename coordT>
coordT&
DetectionPositionPair<coordT>::
timing_pos()
{ return _timing_pos; }

    //! comparison operators
template <typename coordT>
bool
DetectionPositionPair<coordT>::
operator==(const DetectionPositionPair& p) const
{
  // Slightly complicated as we need to be able to cope with reverse order of detectors. If so,
  // the TOF bin should swap as well. However, currently, coordT is unsigned, so timing_pos is
  // always positive so sign reversal can never occur. Below implementation is ok, but
  // generates a compiler warning on many compilers for unsigned.
  // For an unsigned type, we should check
  //    timing_pos() == coordT(0) && p.timing_pos()  == coordT(0)
  // TODO. differentiate between types
  return 
    (pos1() == p.pos1() && pos2() == p.pos2() && timing_pos() == p.timing_pos()) ||
    (pos1() == p.pos2() && pos2() == p.pos1() && timing_pos() == -p.timing_pos());
}

template <typename coordT>
bool
DetectionPositionPair<coordT>::
operator!=(const DetectionPositionPair& d) const
{ return !(*this==d); }

END_NAMESPACE_STIR

