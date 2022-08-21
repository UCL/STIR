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

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

START_NAMESPACE_STIR
template <typename coordT>
DetectionPositionPair<coordT>::
DetectionPositionPair()
  : _timing_pos(0)
{}

template <typename coordT>
DetectionPositionPair<coordT>::
DetectionPositionPair(const DetectionPosition<coordT>& pos1,
                      const DetectionPosition<coordT>& pos2,
                      const int timing_pos)
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
int
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
int&
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
  // the TOF bin should swap as well.
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

