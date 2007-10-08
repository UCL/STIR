//
// $Id$
//
/*
    Copyright (C) 2006 - $Date$, Hammersmith Imanet Ltd
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
  \ingroup modelling

  \brief Implementations of inline functions of class stir::LoganPlot<num_param,num_samples>

  \author Charalampos Tsoumpas

  $Date$
  $Revision$
*/

START_NAMESPACE_STIR

 //! default constructor
LoganPlot::LoganPlot()
{ _matrix_is_stored=false; }

  //! default destructor
LoganPlot::~LoganPlot()
{ }

 //! Create model matrix from plasma data (has to be in appropriate frames)
ModelMatrix<num_param>
LoganPlot::
get_model_matrix(const PlasmaData& plasma_data,const TimeFrameDefinitions& time_frame_definitions,const unsigned int starting_frame)
{    
  if(_matrix_is_stored==false)
    {
      BasicCoordinate<2,int> min_range;
      BasicCoordinate<2,int> max_range;
      min_range[1]=1;  min_range[2]=starting_frame;
      max_range[1]=num_param;  max_range[2]=plasma_data.size();
      IndexRange<2> data_range(min_range,max_range);
      Array<2,float> logan_array(data_range);
      VectorWithOffset<float> time_vector(min_range[2],max_range[2]);
      PlasmaData::const_iterator cur_iter=plasma_data.begin();

      float sum_value=0;
      unsigned int sample_num;
      for(sample_num=1 ; sample_num<starting_frame; ++sample_num, ++cur_iter )
  	  sum_value+=cur_iter->get_plasma_counts_in_kBq()*(time_frame_definitions.get_end_time(sample_num)-time_frame_definitions.get_start_time(sample_num));

      assert(cur_iter==plasma_data.begin()+starting_frame-1);

      for(sample_num=starting_frame ; cur_iter!=plasma_data.end(); ++sample_num, ++cur_iter )
	{
	  sum_value+=cur_iter->get_plasma_counts_in_kBq()*(time_frame_definitions.get_end_time(sample_num)-time_frame_definitions.get_start_time(sample_num));
	  logan_array[1][sample_num]=sum_value;
	  logan_array[2][sample_num]=cur_iter->get_blood_counts_in_kBq();
	  time_vector[sample_num]=0.5*(time_frame_definitions.get_end_time(sample_num)-time_frame_definitions.get_start_time(sample_num)) ;
	}
      assert(sample_num-1==plasma_data.size());
      this->_model_matrix.set_model_array(logan_array);
//    this->_model_matrix.get_sum_model_array();
      this->_model_matrix.set_time_vector(time_vector);
      this->_matrix_is_stored=true;
    }
  return _model_matrix ; 
}


END_NAMESPACE_STIR
