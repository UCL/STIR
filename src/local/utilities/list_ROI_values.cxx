//
// $Id$
//
/*!
  \file
  \ingroup utilities

  \brief Utility programme for getting ROI values

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
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/
#include "stir/utilities.h"
#include "local/stir/eval_buildblock/compute_ROI_values.h"
#if 0
#include "local/stir/Shape/CombinedShape3D.h"
#include "local/stir/Shape/EllipsoidalCylinder.h"
#endif
#include "local/stir/Shape/DiscretisedShape3D.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/ImageProcessor.h"
#include "stir/KeyParser.h"
#include <iostream>
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
  ImageProcessor<3,float>* filter_ptr;
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


#if 0
void
get_bounding_box(
              CartesianCoordinate3D<float>& bounding_box_bottom,
              CartesianCoordinate3D<float>& bounding_box_top,
              const VoxelsOnCartesianGrid<float>& image);

Shape3D*
ask_Shape3D(const CartesianCoordinate3D<float>& bounding_box_bottom,
            const CartesianCoordinate3D<float>& bounding_box_top);


void
get_bounding_box(
                 CartesianCoordinate3D<float>& bounding_box_bottom,
                 CartesianCoordinate3D<float>& bounding_box_top,
                 const VoxelsOnCartesianGrid<float>& image)
{
  CartesianCoordinate3D<float> image_bottom(image.get_min_z(), image.get_min_y(), image.get_min_x());
  CartesianCoordinate3D<float> image_top(image.get_max_z(), image.get_max_y(), image.get_max_x());
  // TODO remove explicit conversion to CartesianCoordinate3D
  bounding_box_bottom = image_bottom * CartesianCoordinate3D<float> (image.get_voxel_size());
  bounding_box_top =image_top * CartesianCoordinate3D<float> (image.get_voxel_size());
}

Shape3D*
ask_Shape3D(const CartesianCoordinate3D<float>& bounding_box_bottom,
            const CartesianCoordinate3D<float>& bounding_box_top)
{
  const CartesianCoordinate3D<float> bounding_box_centre = 
    (bounding_box_bottom + bounding_box_top)/2;
  
  const float xc = 
    ask_num("Centre X coordinate", 
    bounding_box_bottom.x(), bounding_box_top.x(), bounding_box_centre.x());
  
  const float yc = 
    ask_num("Centre Y coordinate", 
    bounding_box_bottom.y(), bounding_box_top.y(), bounding_box_centre.y());
  
  const float zc = 
    ask_num("Centre Z coordinate",     
    bounding_box_bottom.z(), bounding_box_top.z(), bounding_box_centre.z());
  
  const float alpha =
    ask_num("First angle  ",0,180,0);
  const float beta=
    ask_num(" Second angle ",0,180,0);
  const float gamma=
    ask_num(" Third angle",0,180,0);
  
  double  max_len = norm(bounding_box_top - bounding_box_bottom);
  
  const double Rcyl_a = 
    ask_num("Radius a in mm",
    0.,max_len/4,50.);
  
  const double Rcyl_b = 
    ask_num("Radius b in mm",
    0.,max_len/2,100.);
  
  const double Lcyl = 
    ask_num("Length",
    0.,max_len,max_len);
  
  
  
  cerr << "Centre coordinate: (x,y,z) = (" 
    << xc << ", " << yc << ", " << zc  
    << ")" << endl;
  cerr << "Radius_a = " << Rcyl_a << ",Radius_b="<< Rcyl_b <<",Length = " << Lcyl << endl;            
  
  
return new EllipsoidalCylinder (Lcyl,Rcyl_a,Rcyl_b, 
             CartesianCoordinate3D<float>(zc,yc,xc),
             alpha,beta,gamma);
  
}
#endif

END_NAMESPACE_STIR

USING_NAMESPACE_STIR

int
main(int argc, char *argv[])
{
  if(argc!=6 && argc!=4 && argc!=3) 
  {
    cerr<<endl<<"Usage: " << argv[0] << " output_filename data_filename [ ROI_filename.par [min_plane_num max_plane_num]]\n";
    cerr << "Plane numbers start from 1\n";
    cerr << "When ROI_filename.par is not given, the user will be asked for the parameters. "
      "Use this to see what a .par file should look like.\n."<<endl;
    exit(EXIT_FAILURE);
  }


  //SM
  ofstream out (argv[1]);
  const char * const input_file = argv[2];
  if (!out)
  {
    cout<< "Cannot open output file.\n";
    return 1;
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
  out << "Plane number" <<"         Mean "<<"          "<< "Stddev"
      // "            CV"
      <<endl;
  
  for (int i=min_plane_number;i<=max_plane_number;i++)
  {
    out << i+1  
        <<"         "<<values[i].get_mean()
        <<"                 "<< values[i].get_stddev()
        //<< "              " <<values[i].get_CV()
        <<endl;
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