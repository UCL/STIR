//
//
/*
    Copyright (C) 2006 - 2011, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details

  \file
  \ingroup modelling
  \brief Implementations of inline functions of class stir::PatlakPlot
  \author Charalampos Tsoumpas

  \sa PatlakPlot.h, ModelMatrix.h and KineticModel.h

*/


#include "stir/modelling/PatlakPlot.h"
#include "stir/linear_regression.h"

START_NAMESPACE_STIR

void
PatlakPlot::
set_defaults()
{
  base_type::set_defaults();
  this->_blood_data_filename="";
  this->_cal_factor=1.F;
  this->_starting_frame=0;
  this->_time_shift=0.;
  this->_in_correct_scale=false;
  this->_in_total_cnt=false;
}

const char * const 
PatlakPlot::registered_name = "Patlak Plot";

//! default constructor
PatlakPlot::PatlakPlot()
{ 
  this->_matrix_is_stored=false; 
  this->set_defaults();
}

PatlakPlot::~PatlakPlot()   //!< default destructor
{ }

 //! Simply get model matrix if it has been already stored
ModelMatrix<2>
PatlakPlot::
get_model_matrix() const 
{    
  if(_matrix_is_stored==false)
    error("It seems that ModelMatrix has not been set, yet. ");

  return _model_matrix ; 
}

//! Simply set model matrix 
void PatlakPlot::set_model_matrix(ModelMatrix<2> model_matrix)
{
  this->_model_matrix=model_matrix;
  this->_matrix_is_stored=true;
}

 //! Create model matrix from private members
void
PatlakPlot::
create_model_matrix()
{    
  if(_matrix_is_stored==false)
    {
      // Create empty Model matrix. this is a [2 x frames] matrix, that contains Cp(t) and \int{Ct(t)} for each frame (Cp(t): radiotracer concentration on plasma)

      // NOTE: as we are working in time frames, and not discrete time points, Cp(t) is not a value of Cp at a given single time, t, but instead 
      //       it is the integral of Cp on that time frame , \int_{t_start}^{t_end} Cp(t) dt, for each time frame. The same happens with \int{Cp(t)}
      //       All this is handled in the PlasmaData class, and it's not visible here. 

      // BasicCoordinate just changes the indexing range (instead of 0-N, to some min to some max) 
      BasicCoordinate<2,int> min_range;
      BasicCoordinate<2,int> max_range;
      min_range[1]=1;  min_range[2]=this->_starting_frame;
      max_range[1]=2;  max_range[2]=this->_plasma_frame_data.size();
      IndexRange<2> data_range(min_range,max_range);
      Array<2,float> patlak_array(data_range);
      VectorWithOffset<float> time_vector(min_range[2],max_range[2]);
      PlasmaData::const_iterator cur_iter=this->_plasma_frame_data.begin();

      double sum_value=0.;
      unsigned int sample_num;
      // Compute the value of the integral of Cp(t) for frames before the one we want to start applying Patlak to. 
      // Remember that this code requires all frames, from t=0 to be included, otherwise this integral will be wrongly computed. 
      // TODO: do not require the dynamic images to exist to do this integral. 
      for(sample_num=1 ; sample_num<this->_starting_frame; ++sample_num, ++cur_iter )
        sum_value+=cur_iter->get_plasma_counts_in_kBq()*this->_plasma_frame_data.get_time_frame_definitions().get_duration(sample_num);
      
      assert(cur_iter==this->_plasma_frame_data.begin()+this->_starting_frame-1);
      // For each frame that we are interested in, fill the model matrix. 
      for(sample_num=this->_starting_frame ; cur_iter!=this->_plasma_frame_data.end() ; ++sample_num, ++cur_iter )
        {
         sum_value+=cur_iter->get_plasma_counts_in_kBq()*this->_plasma_frame_data.get_time_frame_definitions().get_duration(sample_num);
         // integral of Cp(t)
         patlak_array[1][sample_num]= static_cast<float>(sum_value);
         // Cp(t)
         patlak_array[2][sample_num]=cur_iter->get_plasma_counts_in_kBq();
        // As we will do the reconstruction in un-corrected data, if the plasma data is corrected, we need to undo that 
        if(this->_plasma_frame_data.get_is_decay_corrected())
          {
            const float dec_fact=
              static_cast<float>(decay_correction_factor(this->_plasma_frame_data.get_isotope_halflife(),this->_plasma_frame_data.get_time_frame_definitions().get_start_time(sample_num),
                                      this->_plasma_frame_data.get_time_frame_definitions().get_end_time(sample_num)));
            patlak_array[1][sample_num]/=dec_fact;
            patlak_array[2][sample_num]/=dec_fact;                                                        
            time_vector[sample_num]= static_cast<float>(0.5*(this->_frame_defs.get_end_time(sample_num)+this->_frame_defs.get_start_time(sample_num)));
          }
    }
  if(this->_plasma_frame_data.get_is_decay_corrected())
    warning("Uncorrecting previous decay correction, while putting the plasma_data into the model_matrix.");
  else
    error("plasma_data have not been corrected during the process, which will create wrong results!!!");
  
      assert(sample_num-1==this->_plasma_frame_data.size());
      this->_model_matrix.set_model_array(patlak_array);
      this->_model_matrix.set_time_vector(time_vector);
      // Uncalibrate the ModelMatrix instead of Calibrating all the Dynamic Images. This should make faster the computation.
      // Supposes the images are not calibrated.
      this->_model_matrix.uncalibrate(this->_cal_factor);      
      if(this->_in_total_cnt)
        this->_model_matrix.convert_to_total_frame_counts(this->_frame_defs);
      this->_model_matrix.set_is_in_correct_scale(this->_in_correct_scale);
      this->_model_matrix.threshold_model_array(.000000001F);
      this->_matrix_is_stored=true;
    }
  else 
    warning("ModelMatrix has been already created");
}

Succeeded 
PatlakPlot::set_up()
{
  //  if (base_type::set_up() != Succeeded::yes)
  //    return Succeeded::no;

  this->create_model_matrix();
  if (this->_matrix_is_stored==true)
    return Succeeded::yes;
  else
    return Succeeded::no;
}

void 
PatlakPlot::apply_linear_regression(ParametricVoxelsOnCartesianGrid & par_image, const DynamicDiscretisedDensity & dyn_image) const
{
  if (!this->_in_correct_scale)
    {
#ifndef NDEBUG
      this->_model_matrix.write_to_file("patlak_matrix_not_in_correct_scale.txt");
#endif //NDEBUG
      const DiscretisedDensityOnCartesianGrid <3,float>*  image_cartesian_ptr = 
        dynamic_cast< DiscretisedDensityOnCartesianGrid<3,float>*  > (((dyn_image.get_densities())[0]).get());
      const BasicCoordinate<3,float> this_grid_spacing = image_cartesian_ptr->get_grid_spacing();
      if (dyn_image.get_scanner_default_bin_size()<=0)
        error("PatlakPlot: The dynamic image currently needs to know the Scanner's default_bin_size. Did you set the 'originating system'?");
      this->_model_matrix.scale_model_matrix(this_grid_spacing[2]/dyn_image.get_scanner_default_bin_size());
#ifndef NDEBUG
      this->_model_matrix.write_to_file("patlak_matrix_in_correct_scale.txt");
#endif //NDEBUG
    }
  //  const DynamicDiscretisedDensity & dyn_image=this->_dyn_image;
  // TODO check consistency of time-frame definitions
  const unsigned int num_frames=(this->_frame_defs).get_num_frames();
  unsigned int frame_num;
  unsigned int starting_frame= this->_starting_frame; 
  Array<2,float> patlak_model_array=this->_model_matrix.get_model_array();
  VectorWithOffset<float> patlak_x(starting_frame-1,num_frames-1);
  VectorWithOffset<float> patlak_y(starting_frame-1,num_frames-1); 
  VectorWithOffset<float> weights(starting_frame-1,num_frames-1);

  // Patlak Linear regression is applied to the data in the format:
  // C(t)/Cp(t)=Ki*\int{Cp(t)}/Cp(t)+Vb
  // therefore our "x" value for the regression is \int{Cp(t)}/Cp(t)  (which we know from the model)
  //
  // NOTE: as we are working in time frames, and not discrete time points, Cp(t) is not a value of Cp at a given single time, t, but instead 
  //       it is the integral of Cp on that time frame , \int_{t_start}^{t_end} Cp(t) dt, for each time frame. The same happens with \int{Cp(t)}
  //       All this is handled in the PlasmaData class, and it's not visible here. 
  for(unsigned int frame_num = starting_frame; 
      frame_num<=num_frames ; ++frame_num )
    {      
      patlak_x[frame_num-1]=patlak_model_array[1][frame_num]/patlak_model_array[2][frame_num];
      weights[frame_num-1]=1;                    
    }   
  {  // Do linear_regression for each voxel // for k j i 
    float slope=0.F;
    float y_intersection=0.F;
    float variance_of_slope=0.F;
    float variance_of_y_intersection=0.F;
    float covariance_of_y_intersection_with_slope=0.F;
    float chi_square = 0.F;  
     
    const int min_k_index = dyn_image[1].get_min_index(); 
    const int max_k_index = dyn_image[1].get_max_index();
    for ( int k = min_k_index; k<= max_k_index; ++k)
      {
        const int min_j_index = dyn_image[1][k].get_min_index(); 
        const int max_j_index = dyn_image[1][k].get_max_index();
        for ( int j = min_j_index; j<= max_j_index; ++j)
          {
            const int min_i_index = dyn_image[1][k][j].get_min_index(); 
            const int max_i_index = dyn_image[1][k][j].get_max_index();
            for ( int i = min_i_index; i<= max_i_index; ++i)
              { 
                // Patlak Linear regression is applied to the data in the format:
                // C(t)/Cp(t)=Ki*\int{Cp(t)}/Cp(t)+Vb
                // therefore our "y" value for the regression is C(t)/Cp(t). C(t) is the dynamic image value. 
                // (remember, these are integrals over the time frame, not single values at discrete t) 
                for ( frame_num = starting_frame; 
                      frame_num<=num_frames ; ++frame_num )
                  patlak_y[frame_num-1]=dyn_image[frame_num][k][j][i]/patlak_model_array[2][frame_num];
                // Apply the regression to this pixel
                linear_regression(y_intersection, slope,
                                  chi_square,
                                  variance_of_y_intersection,
                                  variance_of_slope,
                                  covariance_of_y_intersection_with_slope,
                                  patlak_y,
                                  patlak_x,                   
                                  weights);     
                par_image[k][j][i][2]=y_intersection;
                par_image[k][j][i][1]=slope;
              }
          }
      }    
  }
}

void
PatlakPlot::multiply_dynamic_image_with_model_gradient(ParametricVoxelsOnCartesianGrid & par_image,
                                                       const DynamicDiscretisedDensity & dyn_image) const
{
  if (!this->_in_correct_scale)
    {
#ifndef NDEBUG
      this->_model_matrix.write_to_file("patlak_matrix_not_in_correct_scale.txt");
#endif //NDEBUG
      const DiscretisedDensityOnCartesianGrid <3,float>*  image_cartesian_ptr = 
        dynamic_cast< DiscretisedDensityOnCartesianGrid<3,float>*  > (((dyn_image.get_densities())[0]).get());
      const BasicCoordinate<3,float> this_grid_spacing = image_cartesian_ptr->get_grid_spacing();
      if (dyn_image.get_scanner_default_bin_size()<=0)
        error("PatlakPlot: The dynamic image currently needs to know the Scanner's default_bin_size. Did you set the 'originating system'?");
      this->_model_matrix.scale_model_matrix(this_grid_spacing[2]/dyn_image.get_scanner_default_bin_size());
#ifndef NDEBUG
      this->_model_matrix.write_to_file("patlak_matrix_in_correct_scale.txt");
#endif //NDEBUG
    }
  this->_model_matrix.multiply_dynamic_image_with_model(par_image,dyn_image);
}

void
PatlakPlot::multiply_dynamic_image_with_model_gradient_and_add_to_input(ParametricVoxelsOnCartesianGrid & par_image,
                                                                        const DynamicDiscretisedDensity & dyn_image) const
{
  if (!this->_in_correct_scale)
    {
#ifndef NDEBUG
      this->_model_matrix.write_to_file("patlak_matrix_not_in_correct_scale.txt");
#endif //NDEBUG
      const DiscretisedDensityOnCartesianGrid <3,float>*  image_cartesian_ptr = 
        dynamic_cast< DiscretisedDensityOnCartesianGrid<3,float>*  > (((dyn_image.get_densities())[0]).get());
      const BasicCoordinate<3,float> this_grid_spacing = image_cartesian_ptr->get_grid_spacing();
      if (dyn_image.get_scanner_default_bin_size()<=0)
        error("PatlakPlot: The dynamic image currently needs to know the Scanner's default_bin_size. Did you set the 'originating system'?");
      this->_model_matrix.scale_model_matrix(this_grid_spacing[2]/dyn_image.get_scanner_default_bin_size());
#ifndef NDEBUG
      this->_model_matrix.write_to_file("patlak_matrix_in_correct_scale.txt");
#endif //NDEBUG
    }
  this->_model_matrix.multiply_dynamic_image_with_model_and_add_to_input(par_image,dyn_image);
}
// Should be a virtual function declared in the KineticModels or better to the LinearModels
void
PatlakPlot::get_dynamic_image_from_parametric_image(DynamicDiscretisedDensity & dyn_image,
                                                    const ParametricVoxelsOnCartesianGrid & par_image) const
{
  if (!this->_in_correct_scale)
    {
#ifndef NDEBUG
      this->_model_matrix.write_to_file("patlak_matrix_not_in_correct_scale.txt");
#endif //NDEBUG
      const DiscretisedDensityOnCartesianGrid <3,float>*  image_cartesian_ptr = 
        dynamic_cast< DiscretisedDensityOnCartesianGrid<3,float>*  > (((dyn_image.get_densities())[0]).get());
      const BasicCoordinate<3,float> this_grid_spacing = image_cartesian_ptr->get_grid_spacing();
      if (dyn_image.get_scanner_default_bin_size()<=0)
        error("PatlakPlot: The dynamic image currently needs to know the Scanner's default_bin_size. Did you set the 'originating system'?");
      this->_model_matrix.scale_model_matrix(this_grid_spacing[2]/dyn_image.get_scanner_default_bin_size());
#ifndef NDEBUG
      this->_model_matrix.write_to_file("patlak_matrix_in_correct_scale.txt");
#endif //NDEBUG
    }

  this->_model_matrix.multiply_parametric_image_with_model(dyn_image,par_image); 
}

unsigned int
PatlakPlot::get_starting_frame() const 
{
  return this->_starting_frame;
}

unsigned int
PatlakPlot::get_ending_frame() const
{
  return this->get_time_frame_definitions().get_num_frames();
}

TimeFrameDefinitions 
PatlakPlot::get_time_frame_definitions() const 
{
  return this->_frame_defs;
}


void
PatlakPlot::
initialise_keymap()
{
  base_type::initialise_keymap();
  this->parser.add_start_key("Patlak Plot Parameters");
  this->parser.add_key("Blood Data Filename", &this->_blood_data_filename);
  this->parser.add_key("Calibration Factor", &this->_cal_factor);
  this->parser.add_key("Starting Frame", &this->_starting_frame);
  this->parser.add_key("Time Shift", &this->_time_shift);
  this->parser.add_key("In total counts", &this->_in_total_cnt);
  this->parser.add_key("In correct scale", &this->_in_correct_scale);
  this->parser.add_key("Time Frame Definition Filename", &this->_time_frame_definition_filename); 
  this->parser.add_stop_key("end Patlak Plot Parameters");
}

/*! \todo This currently hard-wired F-18 decay for the plasma data */
bool
PatlakPlot::
post_processing()
{
  if (base_type::post_processing() == true)
    return true;

  // read time frame def    
    if (this->_time_frame_definition_filename.size()!=0)
        this->_frame_defs=TimeFrameDefinitions(this->_time_frame_definition_filename);
    else
      {
        error("No Time Frames Definitions available!!!\n ");
        return true;
      }
  // Reading the input function
  if(this->_blood_data_filename=="0")
    {
      warning("You need to specify a file for the input function.");
      return true;
    }
  else
    {
       this->_if_cardiac=false;
       PlasmaData plasma_data_temp;
       plasma_data_temp.read_plasma_data(this->_blood_data_filename);   // The implementation assumes three list file. 
       // TODO have parameter
       warning("Assuming F-18 tracer for half-life!!!");
       plasma_data_temp.set_isotope_halflife(6586.2F);
       plasma_data_temp.shift_time(this->_time_shift);
       this->_plasma_frame_data=plasma_data_temp.get_sample_data_in_frames(this->_frame_defs);
    }
return false;
}

//! Create model matrix from blood frame data
#if 0
ModelMatrix<2>
PatlakPlot::
get_model_matrix(const BloodFrameData& blood_frame_data, const unsigned int starting_frame)
{ 
  assert(starting_frame>0);   
  if(_matrix_is_stored==false)
    {
      BasicCoordinate<2,int> min_range;
      BasicCoordinate<2,int> max_range;
      min_range[1]=1;  min_range[2]=starting_frame;
      max_range[1]=2;  max_range[2]=blood_frame_data.size();
      IndexRange<2> data_range(min_range,max_range);
      Array<2,float> patlak_array(data_range);
      VectorWithOffset<float> time_vector(min_range[2],max_range[2]);
      BloodFrameData::const_iterator cur_iter=blood_frame_data.begin();
    
      float sum_value=0.F;
      unsigned int sample_num;

      for(sample_num=1 ; sample_num<starting_frame; ++sample_num, ++cur_iter )
        {
          const float blood=cur_iter->get_blood_counts_in_kBq();
          const float durat=(cur_iter->get_frame_end_time_in_s()-cur_iter->get_frame_start_time_in_s());
          sum_value+=blood*durat
          *decay_correct_factor(this->_plasma_frame_data.get_isotope_halflife(),
                                cur_iter->get_frame_start_time_in_s(),
                                cur_iter->get_frame_end_time_in_s()) ;
        }
      assert(cur_iter==blood_frame_data.begin()+starting_frame-1);
      for(sample_num=starting_frame ; cur_iter!=blood_frame_data.end(); ++sample_num, ++cur_iter )
        {
          const float blood=cur_iter->get_blood_counts_in_kBq();
          const float durat=(cur_iter->get_frame_end_time_in_s()-cur_iter->get_frame_start_time_in_s());
          sum_value+=blood*durat
            *decay_correct_factor(this->_plasma_frame_data.get_isotope_halflife(),
                                cur_iter->get_frame_start_time_in_s(),
                                cur_iter->get_frame_end_time_in_s()) ;
          // Normalize with the decay correct factor now.
          patlak_array[1][sample_num]=sum_value/decay_correct_factor(this->_plasma_frame_data.get_isotope_halflife(),
                                                             cur_iter->get_frame_start_time_in_s(),
                                                             cur_iter->get_frame_end_time_in_s()) ;
          patlak_array[2][sample_num]=blood;
          time_vector[sample_num]=0.5*(cur_iter->get_frame_start_time_in_s()+cur_iter->get_frame_end_time_in_s()) ;
        }      
      assert(sample_num-1==blood_frame_data.size());  

      this->_model_matrix.set_model_array(patlak_array);
      this->_model_matrix.set_time_vector(time_vector);
      this->_matrix_is_stored=true;
    }
  return _model_matrix ; 
}
#endif

END_NAMESPACE_STIR
