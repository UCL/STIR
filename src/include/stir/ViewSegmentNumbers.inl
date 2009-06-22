/*!
  \file
  \ingroup projdata

  \brief inline implementations for class stir::ViewSegmentNumbers

  \author Kris Thielemans
  \author Sanida Mustafovic
  \author PARAPET project
  
  $Date$

  $Revision$
*/
/*
    Copyright (C) 2000 PARAPET partners
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

ViewSegmentNumbers::ViewSegmentNumbers()
:segment(0),view(0)
  {}

ViewSegmentNumbers::ViewSegmentNumbers( const int view_num,const int segment_num)
    : segment(segment_num),view(view_num)
  {}

int
ViewSegmentNumbers::segment_num() const
{
  return segment;}
int 
ViewSegmentNumbers::view_num() const
{
  return view;}


int&
ViewSegmentNumbers::segment_num() 
{  return segment;}

int& 
ViewSegmentNumbers::view_num() 
{ return view;}

bool 
ViewSegmentNumbers::
operator<(const ViewSegmentNumbers& other) const
{
  return (view< other.view) ||
    ((view == other.view) && (segment > other.segment));
}

bool 
ViewSegmentNumbers::
operator==(const ViewSegmentNumbers& other) const
{
  return (view == other.view) && (segment == other.segment);
}

bool 
ViewSegmentNumbers::
operator!=(const ViewSegmentNumbers& other) const
{
  return !(*this == other);
}
END_NAMESPACE_STIR
