//
// $Id$: $Date$
//

#ifndef __IndexRange2D_H__
#define __IndexRange2D_H__
/*! 
  \file
  \ingroup buildblock
  \brief This file declares the class IndexRange2D.

  \author Kris Thielemans
  \author PARAPET project

  \date    $Date$

  \version $Revision$
*/

#include "IndexRange.h"

START_NAMESPACE_TOMO

/*!
  \ingroup buildblock
  \brief a 'convenience' class for 2D index ranges.

   Provides an easier constructor for regular ranges.
*/
class IndexRange2D : public IndexRange<2>

{
private:
  typedef IndexRange<2> base_type;

public:
  inline IndexRange2D();
  inline IndexRange2D(const IndexRange<2>& range_v);
  inline IndexRange2D(const int min_1, const int max_1,
                      const int min_2, const int max_2);
  inline IndexRange2D(const int length_1, const int length_2);
  // TODO add 2 arg constructor with BasicCoordinate<2,int> min,max
};


END_NAMESPACE_TOMO

#include "IndexRange2D.inl"

#endif
