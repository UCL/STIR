/*!
  \file
  \ingroup projdata

  \brief inline implementations for class ViewSegmentNumbers

  \author Kris Thielemans
  \author Sanida Mustafovic
  \author PARAPET project
  
  $Date$

  $Revision$
*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, IRSL
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
