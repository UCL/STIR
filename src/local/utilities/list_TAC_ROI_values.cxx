//
// $Id$
//
/*
    Copyright (C) 2005- $Date$, Hammersmith Imanet Ltd
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
  \ingroup utilities
  \brief Utility program for getting TAC for ROI values
  \author Kris Thielemans
  \author Charalampos Tsoumpas

  \par Usage:
  \code 
  list_TAC_ROI_values [--CV] output_filename data_filename [ ROI_filename.par ]
  \endcode
  \param output_filename a text file sorted in list form of: | Frame_num | Start Time | End Time | Mean | StdDev | CV |
  \param data_filename should be in ECAT7 format.
  \param --CV is used to output the Coefficient of Variation as well.
  \Note When ROI_filename.par is not given, the user will be asked for the parameters.

  The .par file has the following format
  \verbatim
  ROIValues Parameters :=

  ; give the ROI an (optional) name. Defaults to the empty string.
  ROI name := some name
  ; see Shape3D hierarchy for possible values
  ROI Shape type:=ellipsoid
  ;; ellipsoid parameters here

  ; if more than 1 ROI is desired, you can do this
  next shape :=
  ROI name := some other name
  ROI Shape type:=ellipsoidal cylinder
  ;; parameters here

  number of samples to take for ROI template-z:=1
  number of samples to take for ROI template-y:=1
  number of samples to take for ROI template-x:=1

  ; specify (optional) filter to apply before computing ROI values
  ; see ImageProcessor hierarchy for possible values
  Image Filter type:=None
  End:=
  \endverbatim

  \todo Add the --V option to include the volume information for the sample region. 
  \todo Merge it with the list_ROI_values.cxx utility. 

  $Date$
  $Revision$
*/
#include "stir/utilities.h"
#include "stir/evaluation/compute_ROI_values.h"
#include "stir/Shape/DiscretisedShape3D.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/DynamicDiscretisedDensity.h"
#include "stir/DataProcessor.h"
#include "stir/KeyParser.h"
#include "stir/is_null_ptr.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>

#ifndef STIR_NO_NAMESPACES
using std::cerr;
using std::endl;
using std::ofstream;
#endif


START_NAMESPACE_STIR
//TODO repetition of postfilter.cxx to be able to use its .par file
class ROIValuesParameters : public KeyParser
{
public:
  ROIValuesParameters();
  virtual void set_defaults();
  virtual void initialise_keymap();
  virtual bool post_processing();
  std::vector<shared_ptr<Shape3D> > shape_ptrs;
  std::vector<string> shape_names;
  CartesianCoordinate3D<int> num_samples;
  shared_ptr<DataProcessor<DiscretisedDensity<3,float> > > filter_ptr;
private:
  shared_ptr<Shape3D> current_shape_sptr;
  string current_shape_name;
  void increment_current_shape_num();

  
};

ROIValuesParameters::ROIValuesParameters()
{
  set_defaults();
  initialise_keymap();
}

void ROIValuesParameters::
increment_current_shape_num()
{
  if (!is_null_ptr( current_shape_sptr))
    {
      shape_ptrs.push_back(current_shape_sptr);
      shape_names.push_back(current_shape_name);
      current_shape_sptr = 0;
      current_shape_name = "";
    }
}

void 
ROIValuesParameters::
set_defaults()
{
  shape_ptrs.resize(0);
  shape_names.resize(0);

  filter_ptr = 0;
  current_shape_sptr = 0;
  current_shape_name = "";
  num_samples = CartesianCoordinate3D<int>(1,1,1);
}

void 
ROIValuesParameters::
initialise_keymap()
{
  add_start_key("ROIValues Parameters");
  add_key("ROI name", &current_shape_name);
  add_parsing_key("ROI Shape type", &current_shape_sptr);
  add_key("next shape", KeyArgument::NONE,
	  (KeywordProcessor)&ROIValuesParameters::increment_current_shape_num);
  add_key("number of samples to take for ROI template-z", &num_samples.z());
  add_key("number of samples to take for ROI template-y", &num_samples.y());
  add_key("number of samples to take for ROI template-x", &num_samples.x());
  add_parsing_key("Image Filter type", &filter_ptr);
  add_stop_key("END"); 
}

bool
ROIValuesParameters::
post_processing()
{
  assert(shape_names.size() == shape_ptrs.size());

  if (!is_null_ptr( current_shape_sptr))
    {
      increment_current_shape_num();
    }
  if (num_samples.z()<=0)
    {
      warning("number of samples to take in z-direction should be strictly positive\n");
      return true;
    }
  if (num_samples.y()<=0)
    {
      warning("number of samples to take in y-direction should be strictly positive\n");
      return true;
    }
  if (num_samples.x()<=0)
    {
      warning("number of samples to take in x-direction should be strictly positive\n");
      return true;
    }
  return false;
}

END_NAMESPACE_STIR

USING_NAMESPACE_STIR

int
main(int argc, char *argv[])
{
  bool do_CV=false;
  bool do_V=false;
  const char * const progname = argv[0];

  if (argc>1 && strcmp(argv[1],"--CV")==0)
    {
      do_CV=true;
      --argc; ++argv;
    }
  if (argc>1 && strcmp(argv[1],"--V")==0)
    {
      do_V=true;
      --argc; ++argv;
      if(strcmp(argv[1],"--CV")==0)
      {
	do_CV=true;
	--argc;++argv;
      }
    }
  if(argc<=2 || argc>8)
  {
    cerr<<"\nUsage: " << progname << " \\\n"
	<< "\t[--CV] [--V] output_filename data_filename [ ROI_filename.par ] start_frame_num end_frame_num \n";
    cerr << "Normally, only mean and stddev are listed.\n"
	 << "Use the option --CV to output the Coefficient of Variation as well.\n"
	 << "Use the option --V to output the Total Volume, as well.\n";
    cerr << "Frame number start from start_frame_num and ends to end_frame_num\n";
    cerr << "When ROI_filename.par is not given, the user will be asked for the parameters.\n"
      "Use this to see what a .par file should look like.\n."<<endl;
    exit(EXIT_FAILURE);
  }

  ofstream out (argv[1]);
  const char * const input_file = argv[2];
  if (!out)
  {
    warning("Cannot open output file.\n");
    return EXIT_FAILURE;
  }
  
  const shared_ptr< DynamicDiscretisedDensity >  dyn_image_sptr=
    DynamicDiscretisedDensity::read_from_file(input_file);
  const DynamicDiscretisedDensity & dyn_image = *dyn_image_sptr;

  const unsigned int num_frames=(dyn_image.get_time_frame_definitions()).get_num_frames();
  const unsigned int start_frame_num= argc>=5 ? atoi(argv[4]) : 1 ;
  const unsigned int end_frame_num= argc>=6 ? atoi(argv[5]) : num_frames ;
  ROIValuesParameters parameters;
  if (argc<4)
    parameters.ask_parameters();
  else
    {
      if (parameters.parse(argv[3]) == false)
	exit(EXIT_FAILURE);
    }
  cerr << "Parameters used (aside from names and ROIs):\n\n" << parameters.parameter_info() << endl;
  
  /* This needs a review
  if (parameters.filter_ptr!=0)
    for (unsigned int frame_num;frame_num<=num_frames;frame_num++)
	    parameters.filter_ptr->apply(dyn_image[frame_num]);
  */
  out << input_file << '\n';
  out << std::setw(15) << "ROI "
      << std::setw(10) << "Frame_num " 
      << std::setw(15) << "Start Time "
      << std::setw(15) << "End Time "
      << std::setw(15) << "Mean "
      << std::setw(15) << "Stddev ";
  if (do_CV)
    out << std::setw(15) << "CV";
  if (do_V)
    out << std::setw(15) << "Volume";
  out  <<'\n';
  
  {
    std::vector<shared_ptr<Shape3D> >::const_iterator current_shape_iter =
      parameters.shape_ptrs.begin();
    std::vector<string >::const_iterator current_name_iter =
      parameters.shape_names.begin();
    for (;
	 current_shape_iter != parameters.shape_ptrs.end();
	 ++current_shape_iter, ++current_name_iter)
      { 
	  for (unsigned int frame_num=start_frame_num;frame_num<=end_frame_num;frame_num++)
	  {
	    const float frame_start_time=(dyn_image.get_time_frame_definitions()).get_start_time(frame_num);
	    const float frame_end_time=(dyn_image.get_time_frame_definitions()).get_end_time(frame_num);

	    ROIValues values;
	    values=compute_total_ROI_values(dyn_image[frame_num], **current_shape_iter, parameters.num_samples);
	    out << std::setw(15) << *current_name_iter
		<< std::setw(10) << frame_num
		<< std::setw(15) << frame_start_time
		<< std::setw(15) << frame_end_time
		<< std::setw(15) << values.get_mean()
		<< std::setw(15) << values.get_stddev();
	    if (do_CV)
	      out << std::setw(15) << values.get_CV();
	    if (do_V)
	      out << std::setw(15) << values.get_roi_volume();
	    out <<'\n';
	  }
  
#if 0
	for (VectorWithOffset<ROIValues>::const_iterator iter = values.begin();
	     iter != values.end();
	     iter++)
	  {
	    std::cout << iter->report();
	  }
#endif
      }
  }

  return EXIT_SUCCESS;
}
