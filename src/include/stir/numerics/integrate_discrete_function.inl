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

  \brief Implementations of inline function stir::integrate_discrete_function

  \author Charalampos Tsoumpas

  $Date$
  $Revision$
*/

START_NAMESPACE_STIR

template <typename elemT>
elemT 
integrate_discrete_function(const std::vector<elemT>& t , const std::vector<elemT>& f , const int interpolation_order )
{
  const std::size_t num_samples=f.size();
  elemT integral_result=0;
  assert(num_samples>1);
  if(num_samples!=t.size()) 
    error("integrate_discrete_function requires equal size of the two input vectors!!!");
  
  switch (interpolation_order)
    {
    case 0:
      //Rectangular Formula:
      // If not at the borders apply: (t_next-t_previous)*0.5*f
      // If at the borders apply: (t2-t1)*0.5*f, (tN-TN_previous)*0.5*f
      {	   
	integral_result=f[0]*(t[1]-t[0])*0.5F;
	for (std::size_t i=1;i<num_samples-1;++i)
	  integral_result += f[i]*(t[i+1]-t[i-1])*0.5F;
	integral_result += f[num_samples-1]*(t[num_samples-1]-t[num_samples-2])*0.5F;
      }
      break;
    case 1:
      //trapezoidal
      // Simply apply the formula: (f_next+f)*(t_next-t)*0.5
      {	
	for (std::size_t i=0 ; i<num_samples-1 ; ++i)
		integral_result += (f[i]+f[i+1])*(t[i+1]-t[i])*0.5F;
      }
      break;
    default:
      error("integrate_discrete_function need interpolation order 0 or 1");
    }
  return integral_result;
}

END_NAMESPACE_STIR
