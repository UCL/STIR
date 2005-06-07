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
  \ingroup buildblock
  
  \brief Implementation of some functions missing from std::algorithm
    
  \author Kris Thielemans

  $Date$
  $Revision$
*/
// not nice to have a dependency in buildblock on numerics/norm, but
// we'll bother with that when necessary.
#include "stir/numerics/norm.h"

START_NAMESPACE_STIR

template <class iterT> 
iterT abs_max_element(iterT start, iterT end)
{
  if (start == end)
    return start;
  iterT current_max_iter=start;
  double current_max=norm_squared(*start);
  iterT iter=start; ++iter;

  while(iter != end)
    {
      const double n=norm_squared(*iter);
      if (n>current_max)
	{
	  current_max=n; current_max_iter=iter;
	}
      ++iter;
    }
  return current_max_iter;
}

template <class IterT, class elemT>	
inline 
elemT
sum(IterT start, IterT end, elemT init)
{
  elemT tmp = init;
  for (IterT iter=start; iter!=end; ++iter)
    tmp += *iter;
  return tmp;
}

template <class IterT>	
inline 
typename std::iterator_traits<IterT>::value_type
sum(IterT start, IterT end)
{
  if (start==end)
    {
      typename std::iterator_traits<IterT>::value_type tmp;
      tmp *= 0;
      return tmp;
    }
  return sum(start+1,end,*start);
}

template <class IterT>	
inline 
typename std::iterator_traits<IterT>::value_type
average(IterT start, IterT end)
{
  typename std::iterator_traits<IterT>::value_type tmp =
    sum(start, end);
  tmp /= (end - start);
  return tmp;
}


END_NAMESPACE_STIR

