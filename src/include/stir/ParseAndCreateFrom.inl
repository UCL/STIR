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
  \ingroup densitydata

  \brief  implementation of the stir::ParseAndCreateFrom class for stir:DiscretisedDensity

  \author Kris Thielemans      
*/

#include "stir/VoxelsOnCartesianGrid.h"
//#include "stir/IO/ExamData.h"
#include "stir/CartesianCoordinate3D.h"

START_NAMESPACE_STIR



template <class elemT, class ExamDataT>
DiscretisedDensity<3, elemT>*
ParseAndCreateFrom<DiscretisedDensity<3, elemT>, ExamDataT>::
create(const ExamDataT& exam_data) const
{
  return
    new VoxelsOnCartesianGrid<elemT> (exam_data.get_exam_info_sptr(),
                                      *exam_data.get_proj_data_info_ptr(),
                                      CartesianCoordinate3D<float>(this->get_zoom_z(),
                                                                   this->get_zoom_xy(),
                                                                   this->get_zoom_xy()),
                                      this->get_offset(),
                                      CartesianCoordinate3D<int>(this->get_output_image_size_z(),
                                                                 this->get_output_image_size_xy(),
                                                                 this->get_output_image_size_xy())
                                      );
}

END_NAMESPACE_STIR

