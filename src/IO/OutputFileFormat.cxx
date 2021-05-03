/*
    Copyright (C) 2003- 2009-10-08, Hammersmith Imanet Ltd
    Copyright (C) 2011, Kris Thielemans
    This file is part of STIR.
    SPDX-License-Identifier: Apache-2.0


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

