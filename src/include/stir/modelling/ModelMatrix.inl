//
//
/*
    Copyright (C) 2006 - 2011, Hammersmith Imanet Ltd
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

  \brief Implementations of inline functions of class stir::ModelMatrix

  \author Charalampos Tsoumpas

*/

#include <algorithm>
START_NAMESPACE_STIR

//! default constructor
template <int num_param> 
ModelMatrix<num_param>::ModelMatrix()
{
  // Calibrated ModelMatrix means that the counts are in kBq/ml, while uncalibrated means that it will be to the same units as the reconstructed images.
  this->_is_uncalibrated=false; 
  // Converted ModelMatrix means that it is in total counts in respect to the time_frame_duration, while false means that it will be in mean count rate.
  this->_in_correct_scale=false;
  this->_is_converted_to_total_counts=false;
}

//! default destructor
template <int num_param> 
ModelMatrix<num_param>::~ModelMatrix()
{ }

//! Implementation to read the model matrix
template <int num_param> 
void ModelMatrix<num_param>::read_from_file(const std::string input_string) 
{ 
  std::ifstream data_stream(input_string.c_str()); 
  unsigned int  starting_frame, last_frame;
  if(!data_stream)    
    error("cannot read model matrix from file.\n");    
  else
    {
      data_stream >> starting_frame ;
      data_stream >> last_frame ;
    }
    
  BasicCoordinate<2,int> min_range;
  BasicCoordinate<2,int> max_range;
  min_range[1]=1;  min_range[2]=starting_frame;
  max_range[1]=num_param;  max_range[2]=last_frame;
  IndexRange<2> data_range(min_range,max_range);
  Array<2,float> input_array(data_range);
  while(true)
    {
      for(unsigned int frame_num=starting_frame; frame_num<=last_frame; ++frame_num)
        for(int param_num=1;param_num<=num_param;++param_num)
          data_stream >> input_array[param_num][frame_num] ;
      if(!data_stream) 
        break;               
    }
  this->_model_array=input_array;  // I do not pass info if it is calibrated and if it includes time frame_duration, yet.
}     

//! Implementation to write the model matrix
template <int num_param> 
Succeeded ModelMatrix<num_param>::write_to_file(const std::string output_string) 
{ 

  BasicCoordinate<2,int> model_array_min, model_array_max;
  if(!(this->_model_array).get_regular_range(model_array_min,model_array_max))
    error("Model array has not regular range");
  unsigned int  starting_frame=model_array_min[2], last_frame=model_array_max[2];

  std::ofstream data_stream(output_string.c_str(),std::ios::out); 
  if(!data_stream)    
    {
    warning("ModelMatrix<num_param>::write_to_file: error opening output file %s\n",
      output_string.c_str());
    return Succeeded::no;
    }
  else
    {
      data_stream << starting_frame << " " ;
      data_stream << last_frame << " " ;
    }
    
  // It will be good to assert that there will be no writing error.
  for(unsigned int frame_num=starting_frame; frame_num<=last_frame; ++frame_num)
    {
      data_stream << "\n";
      for(int param_num=1;param_num<=num_param;++param_num)
        data_stream << this->_model_array[param_num][frame_num] << " ";
    }
  data_stream.close();
  return Succeeded::yes;
}  
     

template <int num_param> 
void ModelMatrix<num_param>::
set_model_array(const Array<2,float>& model_array)
{ this->_model_array=model_array ; }

template <int num_param> 
Array<2,float> 
ModelMatrix<num_param>::
get_model_array() const
{  return this->_model_array ; }

template <int num_param> 
const VectorWithOffset<float>
ModelMatrix<num_param>::
get_model_array_sum() const
{ 
  BasicCoordinate<2,int> model_array_min, model_array_max;
  if(!(this->_model_array).get_regular_range(model_array_min,model_array_max))
    error("Model array has not regular range");
  VectorWithOffset<float> sum(model_array_min[1],model_array_max[1]);  
  for(int param_num = model_array_min[1];param_num<=model_array_max[1] ; ++param_num)
    {
      sum[param_num]=0.F;
      for(int frame_num = model_array_min[2];frame_num<=model_array_max[2] ; ++frame_num)
        sum[param_num] += this->_model_array[param_num][frame_num] ; 
    }
  return 
      sum;
}

template <int num_param> 
void ModelMatrix<num_param>::
threshold_model_array(const float threshold_value) 
{ 
  BasicCoordinate<2,int> model_array_min, model_array_max;
  if(!(this->_model_array).get_regular_range(model_array_min,model_array_max))
    error("Model array has not regular range");

  for(int param_num = model_array_min[1];param_num<=model_array_max[1] ; ++param_num)
    for(int frame_num = model_array_min[2];frame_num<=model_array_max[2] ; ++frame_num)
      if(this->_model_array[param_num][frame_num]<=0)
        this->_model_array[param_num][frame_num]=threshold_value;

}

template <int num_param> 
void ModelMatrix<num_param>::
set_if_uncalibrated(const bool is_uncalibrated) 
{  this->_is_uncalibrated=is_uncalibrated; }

template <int num_param> 
void ModelMatrix<num_param>::
set_if_in_correct_scale(const bool in_correct_scale) 
{  this->_in_correct_scale=in_correct_scale; }

template <int num_param> 
void ModelMatrix<num_param>::
uncalibrate(const float cal_factor)
{           
  if(this->_is_uncalibrated)
    warning("ModelMatrix is already uncalibrated, so it will be not re-uncalibrated.");
  else
    {
      BasicCoordinate<2,int> model_array_min, model_array_max;
      if(!(this->_model_array).get_regular_range(model_array_min,model_array_max))
        error("Model array has not regular range");

      for(int param_num = model_array_min[1];param_num<=model_array_max[1] ; ++param_num)
        for(int frame_num = model_array_min[2];frame_num<=model_array_max[2] ; ++frame_num)
          this->_model_array[param_num][frame_num]/=cal_factor;
  
      ModelMatrix<num_param>::set_if_uncalibrated(true);
    }  
}

template <int num_param> 
void ModelMatrix<num_param>::
scale_model_matrix(const float scale_factor) 
{
  if (this->_in_correct_scale)
    warning("ModelMatrix is already scaled, so it will not be re-scaled. ");   
  else
    {
      BasicCoordinate<2,int> model_array_min, model_array_max;
      if(!(this->_model_array).get_regular_range(model_array_min,model_array_max))
        error("Model array has not regular range");
      for(int param_num = model_array_min[1];param_num<=model_array_max[1] ; ++param_num)
        for(int frame_num = model_array_min[2];frame_num<=model_array_max[2] ; ++frame_num)
          this->_model_array[param_num][frame_num]*= scale_factor;
      
      this->_in_correct_scale=true;
    } 
}

template <int num_param> 
void ModelMatrix<num_param>::
convert_to_total_frame_counts(const TimeFrameDefinitions& time_frame_definitions) 
{
  if (ModelMatrix<num_param>::_is_converted_to_total_counts==true)
    warning("ModelMatrix is already converted to total counts, so it will not be re-converted. ");
  else
    {
      BasicCoordinate<2,int> model_array_min, model_array_max;
      if(!(this->_model_array).get_regular_range(model_array_min,model_array_max))
        error("Model array has not regular range");
      for(int param_num = model_array_min[1];param_num<=model_array_max[1] ; ++param_num)
        for(int frame_num = model_array_min[2];frame_num<=model_array_max[2] ; ++frame_num)
          this->_model_array[param_num][frame_num]*= static_cast<float>(time_frame_definitions.get_duration(frame_num));
      
      this->_is_converted_to_total_counts=true;
    }
}

template <int num_param> 
void ModelMatrix<num_param>::
set_time_vector(const VectorWithOffset<float>& time_vector) 
{this->_time_vector=time_vector;}

template <int num_param> 
VectorWithOffset<float>
ModelMatrix<num_param>::
get_time_vector() const
{return this->_time_vector;}

template<int num_param>
void 
ModelMatrix<num_param>::
multiply_dynamic_image_with_model_and_add_to_input(ParametricVoxelsOnCartesianGrid & parametric_image,
                                                   const DynamicDiscretisedDensity & dynamic_image ) const
{
  BasicCoordinate<2,int> model_array_min, model_array_max;
  if(!this->_model_array.get_regular_range(model_array_min,model_array_max))
    error("Model array has not regular range");

  // Assert that the sizes of the one frame of the dynamic image is equal with the parametric image size.
  // ChT::ToDo::Might be better to assert that each of the dimensions sizes with their voxle sizes are equal.
  // Could probably use has_same_characteristics()?
  assert(dynamic_image[1].size_all()==parametric_image.size_all());
  assert(dynamic_image.get_time_frame_definitions().get_num_frames()==static_cast<unsigned int> (model_array_max[2]));
  assert(model_array_max[1]-model_array_min[1]+1==num_param);

  const int min_k_index = dynamic_image[1].get_min_index(); 
  const int max_k_index = dynamic_image[1].get_max_index();
  for ( int k = min_k_index; k<= max_k_index; ++k)
    {
      const int min_j_index = dynamic_image[1][k].get_min_index(); 
      const int max_j_index = dynamic_image[1][k].get_max_index();
      for ( int j = min_j_index; j<= max_j_index; ++j)
        {
          const int min_i_index = dynamic_image[1][k][j].get_min_index(); 
          const int max_i_index = dynamic_image[1][k][j].get_max_index();
          for ( int i = min_i_index; i<= max_i_index; ++i)
            for(int param_num = model_array_min[1];param_num<=model_array_max[1] ; ++param_num)
              {
                float sum_over_frames=0.F;
                for(int frame_num = model_array_min[2];frame_num<=model_array_max[2] ; ++frame_num)
                  sum_over_frames+=this->_model_array[param_num][frame_num]*dynamic_image[frame_num][k][j][i]; 
                parametric_image[k][j][i][param_num]+=sum_over_frames;
              }
        }
    }
}

template<int num_param>
void 
ModelMatrix<num_param>::
multiply_dynamic_image_with_model(ParametricVoxelsOnCartesianGrid & parametric_image,
                                  const DynamicDiscretisedDensity & dynamic_image ) const
{
  std::fill(parametric_image.begin_all(), parametric_image.end_all(), 0.F);
  this->multiply_dynamic_image_with_model_and_add_to_input(parametric_image,dynamic_image );
}

template<int num_param>
void 
ModelMatrix<num_param>::
multiply_parametric_image_with_model_and_add_to_input(DynamicDiscretisedDensity & dynamic_image,  
                                                      const ParametricVoxelsOnCartesianGrid & parametric_image ) const
{
  BasicCoordinate<2,int> model_array_min, model_array_max;
  if(!(this->_model_array).get_regular_range(model_array_min,model_array_max))
    error("Model array does not have a regular range");

  // Assert that the sizes of the one frame of the dynamic image is equal with the parametric image size.
  // ChT::ToDo::Might be better to assert that each of the dimensions sizes with their voxle sizes are equal.
  // Maybe this will be easier if I clone the single images for the two and then compare them.
  assert(dynamic_image[1].size_all()==parametric_image.size_all());
  assert(dynamic_image.get_time_frame_definitions().get_num_frames()==static_cast<unsigned int> (model_array_max[2]));
  assert(model_array_max[1]-model_array_min[1]+1==num_param);

  const int min_k_index = dynamic_image[1].get_min_index(); 
  const int max_k_index = dynamic_image[1].get_max_index();
  for ( int k = min_k_index; k<= max_k_index; ++k)
    {
      const int min_j_index = dynamic_image[1][k].get_min_index(); 
      const int max_j_index = dynamic_image[1][k].get_max_index();
      for ( int j = min_j_index; j<= max_j_index; ++j)
        {
          const int min_i_index = dynamic_image[1][k][j].get_min_index(); 
          const int max_i_index = dynamic_image[1][k][j].get_max_index();
          for ( int i = min_i_index; i<= max_i_index; ++i)
            for(int frame_num = model_array_min[2];frame_num<=model_array_max[2] ; ++frame_num)  
          {
            float sum_over_param=0.F;
            for(int param_num = model_array_min[1];param_num<=model_array_max[1] ; ++param_num)
              sum_over_param+=parametric_image[k][j][i][param_num]*this->_model_array[param_num][frame_num]; 
            dynamic_image[frame_num][k][j][i]=sum_over_param;
          }
        }
    }
}

template<int num_param>
void 
ModelMatrix<num_param>::
multiply_parametric_image_with_model(DynamicDiscretisedDensity & dynamic_image,  
                                     const ParametricVoxelsOnCartesianGrid & parametric_image ) const
{
  std::fill(dynamic_image.begin_all(), dynamic_image.end_all(), 0.F);
  this->multiply_parametric_image_with_model_and_add_to_input(dynamic_image, parametric_image);
}

template<int num_param>
void 
ModelMatrix<num_param>::
normalise_parametric_image_with_model_sum( ParametricVoxelsOnCartesianGrid & parametric_image_out,
                                     const ParametricVoxelsOnCartesianGrid & parametric_image ) const
{
  BasicCoordinate<2,int> model_array_min, model_array_max;
  if(!(this->_model_array).get_regular_range(model_array_min,model_array_max))
    error("Model array has not regular range");

  assert(parametric_image_out.size_all()==parametric_image.size_all());
  assert(model_array_max[1]-model_array_min[1]+1==num_param);

  const int min_k_index = parametric_image.construct_single_density(num_param).get_min_index(); 
  const int max_k_index = parametric_image.construct_single_density(num_param).get_max_index();
  for ( int k = min_k_index; k<= max_k_index; ++k)
    {
      const int min_j_index = (parametric_image.construct_single_density(num_param))[k].get_min_index(); 
      const int max_j_index = (parametric_image.construct_single_density(num_param))[k].get_max_index();
      for ( int j = min_j_index; j<= max_j_index; ++j)
        {
          const int min_i_index = (parametric_image.construct_single_density(num_param))[k][j].get_min_index(); 
          const int max_i_index = (parametric_image.construct_single_density(num_param))[k][j].get_max_index();
          for ( int i = min_i_index; i<= max_i_index; ++i)
            {
              parametric_image_out[k][j][i][1]=parametric_image[k][j][i][1]/((this->get_model_array_sum())[2]);  
              parametric_image_out[k][j][i][2]=parametric_image[k][j][i][2]/((this->get_model_array_sum())[1]);
            }
        }
    }
}
 


END_NAMESPACE_STIR

