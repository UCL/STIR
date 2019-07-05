/*
    Copyright (C) 2019, University College London
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
  \ingroup modelling

  \brief  Definition of the stir::ParseAndCreateFrom class for stir::ParametricDiscretisedDensity

  \author Kris Thielemans
*/

#ifndef __stir_ParseAndCreateParametricDiscretisedDensityFrom_H__
#define __stir_ParseAndCreateParametricDiscretisedDensityFrom_H__

#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/modelling/ParametricDiscretisedDensity.h"
#include "stir/ParseDiscretisedDensityParameters.h"
START_NAMESPACE_STIR

class KeyParser;

//! parse keywords for creating a parametric VoxelsOnCartesianGrid from DynamicProjData etc
/*!
  \ingroup modelling
  \see ParseAndCreateFrom<DiscretisedDensity<3, elemT>, ExamDataT>
*/
template <class elemT, class ExamDataT>
  class ParseAndCreateFrom<ParametricDiscretisedDensity<VoxelsOnCartesianGrid<elemT> >,
                           ExamDataT>
  : public ParseDiscretisedDensityParameters
{
 public:
  typedef ParametricDiscretisedDensity<VoxelsOnCartesianGrid<elemT> > output_type;
  inline
    output_type*
    create(const ExamDataT&) const;
};

END_NAMESPACE_STIR

#include "stir/modelling/ParseAndCreateParametricDiscretisedDensityFrom.inl"
#endif
