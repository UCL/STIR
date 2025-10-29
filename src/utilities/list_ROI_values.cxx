/*
    Copyright (C) 2000 - 2007, Hammersmith Imanet Ltd
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup utilities

  \brief Utility program for getting ROI values

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

  \author Kris Thielemans
*/
#include "stir/utilities.h"
#include "stir/evaluation/compute_ROI_values.h"
#include "stir/Shape/DiscretisedShape3D.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/DataProcessor.h"
#include "stir/KeyParser.h"
#include "stir/is_null_ptr.h"
#include "stir/IO/read_from_file.h"
#include "stir/warning.h"
#include "stir/error.h"
#include "stir/format.h"
#include <iostream>
#include <iomanip>
#include <fstream>
#include <vector>

using std::cerr;
using std::endl;
using std::ofstream;

START_NAMESPACE_STIR
// TODO repetition of postfilter.cxx to be able to use its .par file
class ROIValuesParameters : public KeyParser
{
public:
  ROIValuesParameters();
  virtual void set_defaults();
  virtual void initialise_keymap();
  bool post_processing() override;
  std::vector<shared_ptr<Shape3D>> shape_ptrs;
  std::vector<std::string> shape_names;
  CartesianCoordinate3D<int> num_samples;
  shared_ptr<DataProcessor<DiscretisedDensity<3, float>>> filter_ptr;

private:
  shared_ptr<Shape3D> current_shape_sptr;
  std::string current_shape_name;
  void increment_current_shape_num();
};

ROIValuesParameters::ROIValuesParameters()
{
  set_defaults();
  initialise_keymap();
}

void
ROIValuesParameters::increment_current_shape_num()
{
  if (!is_null_ptr(current_shape_sptr))
    {
      shape_ptrs.push_back(current_shape_sptr);
      shape_names.push_back(current_shape_name);
      current_shape_sptr.reset();
      current_shape_name = "";
    }
}

void
ROIValuesParameters::set_defaults()
{
  shape_ptrs.resize(0);
  shape_names.resize(0);

  filter_ptr.reset();
  current_shape_sptr.reset();
  current_shape_name = "";
  num_samples = CartesianCoordinate3D<int>(1, 1, 1);
}

void
ROIValuesParameters::initialise_keymap()
{
  add_start_key("ROIValues Parameters");
  add_key("ROI name", &current_shape_name);
  add_parsing_key("ROI Shape type", &current_shape_sptr);
  add_key("next shape", KeyArgument::NONE, (KeywordProcessor)&ROIValuesParameters::increment_current_shape_num);
  add_key("number of samples to take for ROI template-z", &num_samples.z());
  add_key("number of samples to take for ROI template-y", &num_samples.y());
  add_key("number of samples to take for ROI template-x", &num_samples.x());
  add_parsing_key("Image Filter type", &filter_ptr);
  add_stop_key("END");
}

bool
ROIValuesParameters::post_processing()
{
  assert(shape_names.size() == shape_ptrs.size());

  if (!is_null_ptr(current_shape_sptr))
    {
      increment_current_shape_num();
    }
  if (num_samples.z() <= 0)
    {
      warning("number of samples to take in z-direction should be strictly positive\n");
      return true;
    }
  if (num_samples.y() <= 0)
    {
      warning("number of samples to take in y-direction should be strictly positive\n");
      return true;
    }
  if (num_samples.x() <= 0)
    {
      warning("number of samples to take in x-direction should be strictly positive\n");
      return true;
    }
  return false;
}

END_NAMESPACE_STIR

USING_NAMESPACE_STIR

int
main(int argc, char* argv[])
{
  bool do_CV = false;
  bool do_V = false;
  bool do_filename = false;
  bool do_max = false;
  bool do_min = false;

  const char* const progname = argv[0];

  while (argc > 1 && strncmp(argv[1], "--", 2) == 0)
    {
      if (strcmp(argv[1], "--min") == 0)
        do_min = true;
      else if (strcmp(argv[1], "--max") == 0)
        do_max = true;
      else if (strcmp(argv[1], "--list-filename") == 0)
        do_filename = true;
      else if (strcmp(argv[1], "--CV") == 0)
        do_CV = true;
      else if (strcmp(argv[1], "--V") == 0)
        do_V = true;
      else
        error(format("Unknown option {}", argv[1]));
      --argc;
      ++argv;
    }

  if (argc != 6 && argc != 5 && argc != 4 && argc != 3)
    {
      cerr << "\nUsage: " << progname << " \\\n"
           << "\t[--CV] [--V] [--list-filename] [--max] [--min] output_filename data_filename [ ROI_filename.par [min_plane_num "
              "max_plane_num]]\n";
      cerr << "Normally, only mean and stddev are listed.\n"
           << "Use the option --CV to output the Coefficient of Variation as well.\n"
           << "Use the option --V to output the Total Volume, as well.\n"
           << "Use the option --list-filename to output the filename as well.\n"
           << "Use the option --max to output the max value as well.\n"
           << "Use the option --min to output the min as well.\n";
      ;
      cerr << "If [min_plane_num] is set to 0 and no [max_plane_num given] then sum of the plane values will be listed.\n";
      cerr << "When ROI_filename.par is not given, the user will be asked for the parameters.\n"
              "Use this to see what a .par file should look like.\n."
           << endl;
      exit(EXIT_FAILURE);
    }

  ofstream out(argv[1]);
  const char* const input_file = argv[2];
  if (!out)
    {
      warning("Cannot open output file.\n");
      return EXIT_FAILURE;
    }

  shared_ptr<DiscretisedDensity<3, float>> image_ptr(read_from_file<DiscretisedDensity<3, float>>(input_file));
  ROIValuesParameters parameters;
  if (argc < 4)
    parameters.ask_parameters();
  else
    {
      if (parameters.parse(argv[3]) == false)
        exit(EXIT_FAILURE);
    }
  cerr << "Parameters used (aside from names and ROIs):\n\n" << parameters.parameter_info() << endl;

  const int min_plane_number = argc == 6 ? atoi(argv[4]) - 1 : image_ptr->get_min_index();
  const int max_plane_number = argc == 6 ? atoi(argv[5]) - 1 : image_ptr->get_max_index();

  const bool by_plane = argc == 5 ? (atoi(argv[4]) != 0) : true;

  if (!is_null_ptr(parameters.filter_ptr))
    parameters.filter_ptr->apply(*image_ptr);
  if (do_filename)
    out << std::setw(15) << "ImageName";
  else
    out << input_file << '\n';

  out << std::setw(15) << "ROI";

  if (by_plane)
    out << std::setw(10) << "Plane_num";
  out << std::setw(15) << "Mean " << std::setw(15) << "Stddev";
  if (do_max)
    out << std::setw(15) << "Max ";
  if (do_min)
    out << std::setw(15) << "Min ";
  if (do_CV)
    out << std::setw(15) << "CV";
  if (do_V)
    out << std::setw(15) << "Volume";
  out << '\n';
  {
    std::vector<shared_ptr<Shape3D>>::const_iterator current_shape_iter = parameters.shape_ptrs.begin();
    std::vector<std::string>::const_iterator current_name_iter = parameters.shape_names.begin();
    for (; current_shape_iter != parameters.shape_ptrs.end(); ++current_shape_iter, ++current_name_iter)
      {
        if (by_plane)
          {
            VectorWithOffset<ROIValues> values;
            compute_ROI_values_per_plane(values, *image_ptr, **current_shape_iter, parameters.num_samples);

            for (int i = min_plane_number; i <= max_plane_number; i++)
              {
                if (do_filename)
                  out << std::setw(15) << input_file;
                out << std::setw(15) << *current_name_iter << std::setw(10) << i + 1 << std::setw(15) << values[i].get_mean()
                    << std::setw(15) << values[i].get_stddev();
                if (do_max)
                  out << std::setw(15) << values[i].get_max();
                if (do_min)
                  out << std::setw(15) << values[i].get_min();
                if (do_CV)
                  out << std::setw(15) << values[i].get_CV();
                if (do_V)
                  out << std::setw(15) << values[i].get_roi_volume();
                out << '\n';
              }
          }
        if (!by_plane)
          {
            ROIValues values;
            values = compute_total_ROI_values(*image_ptr, **current_shape_iter, parameters.num_samples);
            if (do_filename)
              out << std::setw(15) << input_file;
            out << std::setw(15) << *current_name_iter << std::setw(15) << values.get_mean() << std::setw(15)
                << values.get_stddev();
            if (do_max)
              out << std::setw(15) << values.get_max();
            if (do_min)
              out << std::setw(15) << values.get_min();
            if (do_CV)
              out << std::setw(15) << values.get_CV();
            if (do_V)
              out << std::setw(15) << values.get_roi_volume();
            out << '\n';
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
