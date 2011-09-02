//
// $Id$
//
/*!
  \file 
  \ingroup test
  \brief tests parts of the modelling implementation

  \author Charalampos Tsoumpas

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2006- $Date$, Hammersmith Imanet Ltd
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

  
#include "stir/RunTests.h"
#include "stir/modelling/PatlakPlot.h"
#include "stir/modelling/ModelMatrix.h"
#include "stir/modelling/PlasmaData.h"
#include "stir/modelling/ParametricDiscretisedDensity.h"
#include "stir/TimeFrameDefinitions.h"
#include "stir/utilities.h"
#include <boost/shared_array.hpp>

START_NAMESPACE_STIR

/*!
  \ingroup test
  \brief A simple class to test modelling functions.
*/
class modellingTests : public RunTests
{
public:
  explicit modellingTests(const std::string& directory);

  void run_tests();
private:
  //istream& in;
  std::string directory;
  boost::shared_array<char> full_filename_sptr;

  std::string add_directory(const std::string& filename);
};

modellingTests::
modellingTests(const std::string& directory_v) 
  : directory(directory_v),
    full_filename_sptr(new char[directory_v.length() + 100])
{}

string 
modellingTests::
add_directory(const std::string& filename)
{
  strcpy(this->full_filename_sptr.get(), filename.c_str());
  prepend_directory_name(this->full_filename_sptr.get(),this->directory.c_str());
  return std::string(this->full_filename_sptr.get());
}

void modellingTests::run_tests()
{  
  std::cerr << "Testing basic modelling functions..." << std::endl;
  set_tolerance(0.004);
  
  {
      std::cerr << "Testing the reading of PlasmaData ..." << std::endl;

      PlasmaData file_plasma_data, testing_plasma_data;
      file_plasma_data.read_plasma_data(this->add_directory("triple_plasma.if"));
      std::vector<PlasmaSample> this_plasma_blood_plot;
      const PlasmaSample sample_1(0.5F,.999947F,.0999947F);
      const PlasmaSample sample_2(7573.3F,.450739F,.0450739F);
      const PlasmaSample sample_3(30292.2F,.0412893F,.00412893F);
      this_plasma_blood_plot.push_back(sample_1);
      this_plasma_blood_plot.push_back(sample_2);
      this_plasma_blood_plot.push_back(sample_3);
      testing_plasma_data.set_plot(this_plasma_blood_plot);


      PlasmaData::const_iterator cur_iter_1, cur_iter_2;

      for (cur_iter_1=file_plasma_data.begin(), cur_iter_2=testing_plasma_data.begin(); 
	   cur_iter_1!=file_plasma_data.end(), cur_iter_2!=testing_plasma_data.end() ; 
	   ++cur_iter_1, ++cur_iter_2)
	{
	  check_if_equal((*cur_iter_1).get_time_in_s(),(*cur_iter_2).get_time_in_s(), "Check Reading Time of PlasmaData ");
	  check_if_equal((*cur_iter_1).get_plasma_counts_in_kBq(),(*cur_iter_2).get_plasma_counts_in_kBq(), "Check Reading Plasma of PlasmaData ");
	  check_if_equal((*cur_iter_1).get_blood_counts_in_kBq(),(*cur_iter_2).get_blood_counts_in_kBq(), "Check Reading Blood of PlasmaData ");
	}       
  }

  {
      std::cerr << "Testing the reading and writing of the ModelMatrix ..." << std::endl;

      ModelMatrix<2> file_model_matrix, correct_model_matrix;
      file_model_matrix.read_from_file(this->add_directory("model_array.in"));
      file_model_matrix.write_to_file(this->add_directory("model_array.out"));

      BasicCoordinate<2,int> min_range;
      BasicCoordinate<2,int> max_range;
      min_range[1]=1;  min_range[2]=23;
      max_range[1]=2;  max_range[2]=28;
      IndexRange<2> data_range(min_range,max_range);
      Array<2,float> correct_model_array(data_range);
      correct_model_array[1][23]=1;      correct_model_array[2][23]=2;      
      correct_model_array[1][24]=11;     correct_model_array[2][24]=22;     
      correct_model_array[1][25]=111;    correct_model_array[2][25]=222;    
      correct_model_array[1][26]=1111;   correct_model_array[2][26]=2222;   
      correct_model_array[1][27]=11111;  correct_model_array[2][27]=22222;  
      correct_model_array[1][28]=111111; correct_model_array[2][28]=222222; 

      correct_model_matrix.set_model_array(correct_model_array);

      Array<2,float> file_model_array=file_model_matrix.get_model_array();
      Array<2,float> get_correct_model_array=correct_model_matrix.get_model_array();

      for (unsigned int param_num=1;param_num<=2;++param_num)
	for(unsigned int frame_num=23;frame_num<=28;++frame_num)
	  {
	    check_if_equal(file_model_array[param_num][frame_num],get_correct_model_array[param_num][frame_num],"Check ModelMatrix reading. ");  
	    check_if_equal(file_model_array[param_num][frame_num],correct_model_array[param_num][frame_num],"Check ModelMatrix reading. ");  
	  }
  }
  {
    // This tests uses the results from the Mathematica. The used plasma and frame files are parts of the t00196 scan.
    std::cerr << "\nTesting the sampling of PlasmaData into frames ..." << std::endl;  
    PlasmaData file_plasma_data, testing_plasma_data;
    file_plasma_data.read_plasma_data(this->add_directory("plasma.if"));
    std::vector<PlasmaSample> this_plasma_blood_plot;
    TimeFrameDefinitions time_frame_def(this->add_directory("time.fdef")); 
    file_plasma_data.set_isotope_halflife(6586.2F);
    PlasmaData sample_plasma_data_in_frames = file_plasma_data.get_sample_data_in_frames(time_frame_def);

    const PlasmaSample sample_17(1, 11.4776, 10.7832);  const PlasmaSample sample_18(1, 10.7523, 10.1135);  const PlasmaSample sample_19(1, 10.0841, 9.50239);
    const PlasmaSample sample_20(1, 9.24207, 8.7949);   const PlasmaSample sample_21(1, 8.39741, 8.04141);  const PlasmaSample sample_22(1, 7.74369, 7.36121);
    const PlasmaSample sample_23(1, 7.18224, 6.78764);  const PlasmaSample sample_24(1, 6.67699, 6.3266);   const PlasmaSample sample_25(1, 6.23402, 5.93635);
    const PlasmaSample sample_26(1, 5.8495, 5.593);     const PlasmaSample sample_27(1, 5.50858, 5.29071);  const PlasmaSample sample_28(1, 5.19509, 5.02458);
    
    TimeFrameDefinitions plasma_fdef=sample_plasma_data_in_frames.get_time_frame_definitions();

    this_plasma_blood_plot.push_back(sample_17);    this_plasma_blood_plot.push_back(sample_18);    this_plasma_blood_plot.push_back(sample_19);
    this_plasma_blood_plot.push_back(sample_20);    this_plasma_blood_plot.push_back(sample_21);    this_plasma_blood_plot.push_back(sample_22);    
    this_plasma_blood_plot.push_back(sample_23);    this_plasma_blood_plot.push_back(sample_24);    this_plasma_blood_plot.push_back(sample_25);    
    this_plasma_blood_plot.push_back(sample_26);    this_plasma_blood_plot.push_back(sample_27);    this_plasma_blood_plot.push_back(sample_28);    

    testing_plasma_data.set_plot(this_plasma_blood_plot);
    testing_plasma_data.set_isotope_halflife(6586.2F);
    testing_plasma_data.decay_correct_PlasmaData();
    PlasmaData::const_iterator cur_iter_1, cur_iter_2;
    
    for (cur_iter_1=sample_plasma_data_in_frames.begin()+16, cur_iter_2=testing_plasma_data.begin(); 
	   cur_iter_1!=sample_plasma_data_in_frames.end(), cur_iter_2!=testing_plasma_data.end() ; 
	   ++cur_iter_1, ++cur_iter_2)
	{	  
	  check_if_equal((*cur_iter_1).get_plasma_counts_in_kBq(),(*cur_iter_2).get_plasma_counts_in_kBq(),"Check Plasma when sampling PlasmaData into frames");
	  check_if_equal((*cur_iter_1).get_blood_counts_in_kBq(),(*cur_iter_2).get_blood_counts_in_kBq(),"Check Blood when sampling PlasmaData into frames");
	}   
    assert(time_frame_def.get_num_frames()==plasma_fdef.get_num_frames());
    for (unsigned int frame_num=17 ; frame_num<=time_frame_def.get_num_frames() && frame_num<=plasma_fdef.get_num_frames(); ++frame_num)
      {
	check_if_equal(time_frame_def.get_start_time(frame_num),plasma_fdef.get_start_time(frame_num),"Check start time when sampling PlasmaData into frames");
	check_if_equal(time_frame_def.get_duration(frame_num),plasma_fdef.get_duration(frame_num),"Check duration when sampling PlasmaData into frames");
	check_if_equal(time_frame_def.get_end_time(frame_num),plasma_fdef.get_end_time(frame_num),"Check duration when sampling PlasmaData into frames");
      }    
      std::cerr << "\nTesting the creation of Model Matrix based on Plasma Data..." << std::endl;     
      PatlakPlot patlak_plot;      
      const unsigned int starting_frame=23;
      ModelMatrix<2> stir_model_matrix=(patlak_plot.get_model_matrix(sample_plasma_data_in_frames,time_frame_def,starting_frame));
      ModelMatrix<2> mathematica_model_matrix;
      mathematica_model_matrix.read_from_file(this->add_directory("math_model_matrix.in"));
      stir_model_matrix.uncalibrate(10);
      //     stir_model_matrix.convert_to_total_frame_counts(time_frame_def);
      Array<2,float> stir_model_array=stir_model_matrix.get_model_array();
      Array<2,float> mathematica_model_array=mathematica_model_matrix.get_model_array();

      for(unsigned int frame_num=23;frame_num<=28;++frame_num)
	{
	  check_if_equal(mathematica_model_array[1][frame_num]/10.F,stir_model_array[1][frame_num],"Check _model_array-1st column in ModelMatrix");
	  check_if_equal(mathematica_model_array[2][frame_num]/10.F,stir_model_array[2][frame_num],"Check _model_array-2nd column in ModelMatrix");
	}
  }

}


END_NAMESPACE_STIR

USING_NAMESPACE_STIR

int main(int argc, char **argv)
{
  if (argc != 2)
  {
    cerr << "Usage : " << argv[0] << " <directory-name-for-input-files>\n";
    return EXIT_FAILURE;
  }
  modellingTests tests(argv[1]);
  tests.run_tests();
  return tests.main_return_value();
}
