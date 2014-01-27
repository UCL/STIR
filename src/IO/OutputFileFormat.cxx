/*
    Copyright (C) 2003- 2009-10-08, Hammersmith Imanet Ltd
    Copyright (C) 2011, Kris Thielemans
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
  \ingroup IO
  
  \brief  Instantiations of the stir::OutputFileFormat class 
  \author Kris Thielemans      
*/

#include "stir/IO/OutputFileFormat.txx"
#include "stir/DiscretisedDensity.h"
#include "stir/DynamicDiscretisedDensity.h"
#include "stir/modelling/ParametricDiscretisedDensity.h" 
#include "stir/modelling/KineticParameters.h" 

#ifdef _MSC_VER
#pragma warning (disable : 4661)
#endif

START_NAMESPACE_STIR

 
template class OutputFileFormat<DiscretisedDensity<3,float> >; 
template class OutputFileFormat<DynamicDiscretisedDensity >; 
template class OutputFileFormat<ParametricVoxelsOnCartesianGrid >;  

END_NAMESPACE_STIR

