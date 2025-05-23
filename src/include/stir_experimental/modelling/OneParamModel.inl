//
//
/*
    Copyright (C) 2006 - 2007, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details

  \file
  \ingroup modelling

  \brief Implementations of inline functions of class stir::PatlakPlot<num_param,num_samples>

  \author Charalampos Tsoumpas

*/

START_NAMESPACE_STIR

//! default constructor
OneParamModel::OneParamModel()
{ _matrix_is_stored=false; }

OneParamModel::OneParamModel(const int starting_frame, const int last_frame)
{
  this->_matrix_is_stored=false; 
  this->_starting_frame=starting_frame;
  this->_last_frame=last_frame;
}

//! default destructor
OneParamModel::~OneParamModel()
{ }

//! Create a unit model matrix for a single frame and single parameter 
ModelMatrix<1> 
OneParamModel::
get_unit_matrix(const int starting_frame, const int last_frame)
{    
  if(_matrix_is_stored==false)
    {
      this->_starting_frame=starting_frame;
      this->_last_frame=last_frame;
      BasicCoordinate<2,int> min_range;
      BasicCoordinate<2,int> max_range;
      min_range[1]=1;  min_range[2]=this->_starting_frame;
      max_range[1]=1;  max_range[2]=this->_last_frame;
      IndexRange<2> data_range(min_range,max_range);
      Array<2,float> unit_array(data_range);
      
      for(int frame_num=this->_starting_frame ; frame_num<=this->_last_frame; ++frame_num) 
	unit_array[1][frame_num]=1.F;

      _unit_matrix.set_model_array(unit_array);
      _matrix_is_stored=true;
    }
  return _unit_matrix ; 
}

END_NAMESPACE_STIR
