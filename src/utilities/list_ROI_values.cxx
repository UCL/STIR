//
// $Id$
//
/*!
  \file
  \ingroup utilities

  \brief Utility program for getting ROI values

  The .par file has the following format
  \verbatim
  ROIValues Parameters :=
  ; see Shape3D hierarchy for possible values
  ROI Shape type:=None
  number of samples to take for ROI template-z:=1
  number of samples to take for ROI template-y:=1
  number of samples to take for ROI template-x:=1
  ; see ImageProcessor hierarchy for possible values
  Image Filter type:=None
  End:=
  \endverbatim

  \author Kris Thielemans
  $Date$
  $Revision$

*/
/*
    Copyright (C) 2000- $Date$, Hammersmith Imanet Ltd
    See STIR/LICENSE.txt for details
*/
#include "stir/utilities.h"
#include "stir/evaluation/compute_ROI_values.h"
#include "stir/Shape/DiscretisedShape3D.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/ImageProcessor.h"
#include "stir/KeyParser.h"
#include <iostream>
#include <iomanip>
#include <fstream>

#ifndef STIR_NO_NAMESPACES
using std::cerr;
using std::endl;
using std::cout;
using std::ofstream;
#endif


START_NAMESPACE_STIR
//TODO repetition of postfilter.cxx to be able to use its .par file
class ROIValuesParameters 
{
public:
  ROIValuesParameters();
  Shape3D * shape_ptr;
  CartesianCoordinate3D<int> num_samples;
  shared_ptr<ImageProcessor<3,float> > filter_ptr;
public:
  KeyParser parser;
  
};

ROIValuesParameters::ROIValuesParameters()
{
  filter_ptr = 0;
  shape_ptr = 0;
  num_samples = CartesianCoordinate3D<int>(1,1,1);
  parser.add_start_key("ROIValues Parameters");
  parser.add_parsing_key("ROI Shape type", &shape_ptr);
  parser.add_key("number of samples to take for ROI template-z", &num_samples.z());
  parser.add_key("number of samples to take for ROI template-y", &num_samples.y());
  parser.add_key("number of samples to take for ROI template-x", &num_samples.x());
  parser.add_parsing_key("Image Filter type", &filter_ptr);
  parser.add_stop_key("END"); 
}


END_NAMESPACE_STIR

USING_NAMESPACE_STIR

int
main(int argc, char *argv[])
{
  bool do_CV=false;
  const char * const progname = argv[0];

  if (argc>1 && strcmp(argv[1],"--CV")==0)
    {
      do_CV=true;
      --argc; ++argv;
    }
  if(argc!=6 && argc!=4 && argc!=3) 
  {
    cerr<<"\nUsage: " << progname << " \\\n"
	<< "\t[--CV] output_filename data_filename [ ROI_filename.par [min_plane_num max_plane_num]]\n";
    cerr << "Normally, only mean and stddev are listed.\n"
	 << "Use the option --CV to output the Coefficient of Variation as well.\n";
    cerr << "Plane numbers start from 1\n";
    cerr << "When ROI_filename.par is not given, the user will be asked for the parameters.\n"
      "Use this to see what a .par file should look like.\n."<<endl;
    exit(EXIT_FAILURE);
  }


  ofstream out (argv[1]);
  const char * const input_file = argv[2];
  if (!out)
  {
    cout<< "Cannot open output file.\n";
    return EXIT_FAILURE;
  }
  
  
  shared_ptr<DiscretisedDensity<3,float> > image_ptr =
    DiscretisedDensity<3,float>::read_from_file(input_file);

  ROIValuesParameters parameters;
  if (argc<4)
    parameters.parser.ask_parameters();
  else
    parameters.parser.parse(argv[3]);
  cerr << "Parameters used:\n" << parameters.parser.parameter_info() << endl;


  const int min_plane_number = 
    argc==6 ? atoi(argv[4])-1 : image_ptr->get_min_index();
  const int max_plane_number = 
    argc==6 ? atoi(argv[5])-1 : image_ptr->get_max_index();

#if 0
  CartesianCoordinate3D<float> bounding_box_bottom;
  CartesianCoordinate3D<float> bounding_box_top;
  get_bounding_box(bounding_box_bottom, bounding_box_top, *image_ptr);
  Shape3D* shape_ptr = ask_Shape3D(bounding_box_bottom, bounding_box_top);
#endif  
  
  VectorWithOffset<ROIValues> values;
  
  if (parameters.filter_ptr!=0)
    parameters.filter_ptr->apply(*image_ptr);

  compute_ROI_values_per_plane(values, *image_ptr, *parameters.shape_ptr, parameters.num_samples);
  
  out << input_file << endl;
  out << std::setw(10) << "Plane num" 
      << std::setw(15) << "Mean "
      << std::setw(15) << "Stddev";
  if (do_CV)
    out << std::setw(15) << "CV";
  out  <<'\n';
  
  for (int i=min_plane_number;i<=max_plane_number;i++)
  {
    out << std::setw(10) << i+1  
        << std::setw(15) << values[i].get_mean()
        << std::setw(15) << values[i].get_stddev();
    if (do_CV)
      out << std::setw(15) << values[i].get_CV();
    out <<'\n';
  }
  
#if 0
  for (VectorWithOffset<ROIValues>::const_iterator iter = values.begin();
  iter != values.end();
  iter++)
  {
    cout << iter->report();
  }
#endif

  return EXIT_SUCCESS;
}
