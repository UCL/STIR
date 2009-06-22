//
// $Id$
//

/*!
  \file
  \ingroup projdata
  \brief Implementation of inline methods of class stir::DetectionPosition
  \author Kris Thielemans
  $Date$
  $Revision$
*/
/*
    Copyright (C) 2002- $Date$, Hammersmith Imanet Ltd
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
DetectionPosition<coordT>::
DetectionPosition(const coordT tangential_coord,
  	                   const coordT axial_coord, 
			   const coordT radial_coord)
  : tangential(tangential_coord),
    axial(axial_coord),
    radial(radial_coord)
{}

template <typename coordT>
coordT
DetectionPosition<coordT>::
tangential_coord()  const
{ return tangential; }

template <typename coordT>
coordT
DetectionPosition<coordT>::
axial_coord()const
{ return axial;}

template <typename coordT>
coordT
DetectionPosition<coordT>::
radial_coord()const
{ return radial;}

template <typename coordT>
coordT&
DetectionPosition<coordT>::
tangential_coord()
{ return tangential;}

template <typename coordT>
coordT&
DetectionPosition<coordT>::
axial_coord()
{ return axial;}

template <typename coordT>
coordT&
DetectionPosition<coordT>::
radial_coord()
{ return radial;} 

    //! comparison operators
template <typename coordT>
bool
DetectionPosition<coordT>::
operator==(const DetectionPosition& d) const
{
  return 
    tangential == d.tangential &&
    axial == d.axial &&
    radial == d.radial;
}

template <typename coordT>
bool
DetectionPosition<coordT>::
operator!=(const DetectionPosition& d) const
{ return !(*this==d); }

END_NAMESPACE_STIR

