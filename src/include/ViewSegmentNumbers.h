//
// $Id$: $Date$
//

// First version : Kris Thielemans 03/99


#ifndef __ViewSegmentNumbers_h__
#define __ViewSegmentNumbers_h__

class ViewSegmentNumbers
{
public:
  int segment_num;
  int view_num;

  ViewSegmentNumbers( const int view_num,const int segment_num)
    : segment_num(segment_num),
      view_num(view_num)
  {}

  // comparison operator, only useful for sorting
  // order : (0,1) < (0,-1) < (1,1) ...
  // KT 15/03/99 new
  bool operator<(const ViewSegmentNumbers& other) const
  {
    return (view_num < other.view_num) ||
      ((view_num == other.view_num) && (segment_num > other.segment_num));
  }
};

#endif
