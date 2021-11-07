/*
    Copyright (C) 2019, University College London
    This file is part of STIR. 
 
    SPDX-License-Identifier: Apache-2.0 
 
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
