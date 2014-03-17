/*!
  \file
  \ingroup evaluation

  \brief Definition of class stir::ROIValues

  \author Damiano Belluzzo
  \author Kris Thielemans
  \author PARAPET project
*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- 2009, Hammersmith Imanet Ltd
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

#ifndef __stir_evaluation_ROIValues__H__
#define __stir_evaluation_ROIValues__H__

#include "stir/common.h"

#include <string>
#include <iostream>
#include <algorithm>
#ifndef STIR_NO_NAMESPACES
using std::string;
#endif

START_NAMESPACE_STIR


/*!
  \ingroup evaluation
  \brief A class to store and get results of an ROI calculation.
  
  This class stores the volume of the ROI (in cubic mm), the integral over the ROI of the functions and
  its square and the min and max values in the ROI. These values are used to compute mean, 
  standard deviation and coefficient of variance.
*/
class ROIValues
{

public:

  ROIValues()
    {init();};

  ROIValues(float roi_volume, float integral, float integral_of_square, float min_value, float max_value)
    : roi_volume(roi_volume), integral(integral), integral_of_square(integral_of_square), 
      min_value(min_value), max_value(max_value)
    {
      update();
    };

  //! Combine the ROI values appropriately
  ROIValues operator+= (const ROIValues &iv)	
    {
      roi_volume += iv.roi_volume;
      integral += iv.integral;
      integral_of_square += iv.integral_of_square;


#ifndef STIR_NO_NAMESPACES
      min_value = std::min(min_value, iv.min_value);
      max_value = std::max(max_value, iv.max_value);
#else
      min_value = min(min_value, iv.min_value);
      max_value = max(max_value, iv.max_value);
#endif

      update();
      return *this;
    };

  //! Return a string with all info, one per line
  string report() const;

  //! Total valume (in mm^3)
  float get_roi_volume() const
    { return roi_volume; }
  //! Sum of elements times voxel volume
  float get_integral() const
    { return integral; }
  //! Sum of squares times voxel volume
  float get_integral_of_square() const
    { return integral_of_square; }
  //! Mean value
  float get_mean() const
    { return mean_value; }
  //! Variance
  float get_variance() const
    { return variance_value; }
  //! Standard deviation
  float get_stddev() const
    {return std_value; }
  //! Coefficient of Variance =stddev/mean)
  float get_CV() const
    {return std_value/mean_value; }
  //! Minimum value in the ROI
  float get_min() const 
    { return min_value; }
  //! Maximum value in the ROI
  float get_max() const 
    { return max_value; }


//friend ostream &operator <<( ostream &stream, ROIValues val);

private:

  float roi_volume;
  float integral;
  float integral_of_square;

  float mean_value;
  float variance_value;
  float std_value;

  float min_value;
  float max_value;
  
  void init();
  void update();
};

END_NAMESPACE_STIR

#endif
