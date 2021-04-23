//
//

/*!
  \file
  \ingroup projdata
  \brief Implementation of inline methods of class stir::DetectionPositionPair
  \author Kris Thielemans
*/
/*
    Copyright (C) 2002- 2009, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/

START_NAMESPACE_STIR
template <typename coordT>
DetectionPositionPair<coordT>::
DetectionPositionPair()
{}

template <typename coordT>
DetectionPositionPair<coordT>::
DetectionPositionPair(const DetectionPosition<coordT>& pos1,
                      const DetectionPosition<coordT>& pos2)
  : p1(pos1), p2(pos2)
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
DetectionPosition<coordT>&
DetectionPositionPair<coordT>::
pos1()
{ return p1; }

template <typename coordT>
DetectionPosition<coordT>&
DetectionPositionPair<coordT>::
pos2()
{ return p2; }

    //! comparison operators
template <typename coordT>
bool
DetectionPositionPair<coordT>::
operator==(const DetectionPositionPair& p) const
{
  return 
    (pos1() == p.pos1() && pos2() == p.pos2()) ||
    (pos1() == p.pos2() && pos2() == p.pos1())	;
}

template <typename coordT>
bool
DetectionPositionPair<coordT>::
operator!=(const DetectionPositionPair& d) const
{ return !(*this==d); }

END_NAMESPACE_STIR

