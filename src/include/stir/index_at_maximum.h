//
// $Id$
//
/*
    Copyright (C) 2005- $Date$, Hammersmith Imanet Ltd
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
/*!
  \file
  \ingroup array

  \brief Declaration of stir:index_at_maximum()

  \author Kris Thielemans

  $Date$
  $Revision$
*/

#include "stir/VectorWithOffset.h"

START_NAMESPACE_STIR

/*! \ingroup array
  \brief Finds the index where the maximum occurs.

  If the maximum occurs more than once, the smallest index is returned.

 If the vector is empty, the function returns 0;
*/

template <class elemT>
int index_at_maximum(const VectorWithOffset<elemT>& v)
{
  if (v.size() == 0)
    return 0;

  int index_at_max=v.get_min_index();
  elemT max_value=v[index_at_max];
  for (int index=v.get_min_index(); index<=v.get_max_index(); ++index)
    {
      const elemT value = v[index];
      if (value>max_value)
	{
	  index_at_max=index;
	  max_value=value;
	}
    }
  return index_at_max;
}

END_NAMESPACE_STIR
