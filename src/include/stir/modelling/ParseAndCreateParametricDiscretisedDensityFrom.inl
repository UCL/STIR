/*
    Copyright (C) 2019, University College London
    This file is part of STIR. 
 
    SPDX-License-Identifier: Apache-2.0 
 
    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup densitydata

  \brief  implementation of the stir::ParseAndCreateFrom class for stir:ParametricDiscretisedDensity

  \author Kris Thielemans      
*/

#include "stir/VoxelsOnCartesianGrid.h"
//#include "stir/IO/ExamData.h"
#include "stir/CartesianCoordinate3D.h"

START_NAMESPACE_STIR


template <class elemT, class ExamDataT>
ParametricDiscretisedDensity<VoxelsOnCartesianGrid<elemT> >*
ParseAndCreateFrom<ParametricDiscretisedDensity<VoxelsOnCartesianGrid<elemT> >, ExamDataT>::
create(const ExamDataT& exam_data) const
{

    return
      new ParametricDiscretisedDensity<VoxelsOnCartesianGrid<elemT> >
      (VoxelsOnCartesianGrid<elemT>
       (exam_data.get_exam_info_sptr(),
        *exam_data.get_proj_data_info_sptr(),
        CartesianCoordinate3D<float>(this->get_zoom_z(),
                                     this->get_zoom_xy(),
                                     this->get_zoom_xy()),
        this->get_offset(),
        CartesianCoordinate3D<int>(this->get_output_image_size_z(),
                                   this->get_output_image_size_xy(),
                                   this->get_output_image_size_xy())
        )
       );
}

END_NAMESPACE_STIR

