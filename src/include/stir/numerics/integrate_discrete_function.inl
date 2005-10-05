//
// $Id$
//
/*
    Copyright (C) 2004 - $Date$, Hammersmith Imanet Ltd
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

  \file
  \ingroup numerics

  \brief Implementations of inline functions of linear_integral

  \author Charalampos Tsoumpas

  $Date$
  $Revision$
*/

START_NAMESPACE_STIR

float 
linear_integral(std::vector<float> f , std::vector<float> t , int approx)
{
	std::vector<float>::const_iterator cur_iter_f=f.begin();
 	std::vector<float>::const_iterator cur_iter_t=t.begin();
	float integral_result=0;

	if (approx==0)
{
	integral_result=-(*cur_iter_f)*(*cur_iter_t);
	integral_result+=*cur_iter_f*(*(cur_iter_t+1));

	for (++cur_iter_f , ++cur_iter_t ; 
	 cur_iter_f!=(f.end()-1) , cur_iter_t!=(t.end()-1) ; 
       ++cur_iter_f , ++cur_iter_t)
	integral_result += *cur_iter_f*(*(cur_iter_t+1)-(*(cur_iter_t-1)))*0.5;
	
	++cur_iter_f; ++cur_iter_t;

	integral_result += *cur_iter_f*(*(cur_iter_t)-(*(cur_iter_t-1)));
}
	if (approx==1)
{	
	for (++cur_iter_f , ++cur_iter_t ; 
	 cur_iter_f!=(f.end()-1) , cur_iter_t!=(t.end()-1) ; 
      ++cur_iter_f , ++cur_iter_t)
	integral_result += (*cur_iter_f+(*(cur_iter_f+1)))*(*(cur_iter_t+1)-(*(cur_iter_t)))*0.5;
}
	return integral_result;
}

END_NAMESPACE_STIR
