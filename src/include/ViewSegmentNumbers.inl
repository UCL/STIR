/*!
  \file
  \ingroup buildblock

  \brief inline implementations for class ViewSegmentNumbers

  \author Kris Thielemans
  \author Sanida Mustafovic
  \author PARAPET project
  
  \date $Date$

  \version $Revision$
*/


START_NAMESPACE_TOMO

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
END_NAMESPACE_TOMO
