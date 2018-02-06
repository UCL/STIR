//
//

#ifndef __IndexRange3D_H__
#define __IndexRange3D_H__
/*! 
  \file
  \ingroup Array
  
  \brief This file declares the class stir::IndexRange3D.

  \author Sanida Mustafovic
  \author Kris Thielemans
  \author PARAPET project


*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2009, Hammersmith Imanet Ltd
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

#include "stir/IndexRange.h"

START_NAMESPACE_STIR


/*! 
   \ingroup Array
   \brief a 'convenience' class for 3D index ranges.
   Provides an easier constructor for regular ranges.
*/
class IndexRange3D : public IndexRange<3>

{
private:
  typedef IndexRange<3> base_type;

public:
  inline IndexRange3D();
  inline IndexRange3D(const IndexRange<3>& range_v);
  inline IndexRange3D(const int min_1, const int max_1,
                      const int min_2, const int max_2,
		      const int min_3, const int max_3);
  inline IndexRange3D(const int length_1, const int length_2, const int length_3);
  
  // TODO add 2 arg constructor with BasicCoordinate<2,int> min,max
};


END_NAMESPACE_STIR

#include "stir/IndexRange3D.inl"

#endif
