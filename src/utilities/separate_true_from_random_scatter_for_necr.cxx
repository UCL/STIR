/*
Copyright 2017 ETH Zurich, Institute of Particle Physics and Astrophysics

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

        http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/

/*!
  \file
  \ingroup utilities

  \brief This program gets a projection data file of a mouse/rat scatter phantom measured according to NEMA NU 4.
          It seperates number of true events from number of random+scatter events.

  \author Parisa Khateri

*/

#include "stir/ProjData.h"
#include "stir/IO/interfile.h"
#include "stir/utilities.h"
#include "stir/Bin.h"
#include <math.h>
#include <fstream>
#include <iostream>
#include "stir/ProjDataFromStream.h"
#include "stir/Sinogram.h"
#include "stir/IO/read_from_file.h"
#include "stir/SegmentByView.h"
#include "stir/ProjDataInterfile.h"
#include "stir/ProjDataInfo.h"
#include <algorithm>
#include "stir/error.h"

using std::cerr;

USING_NAMESPACE_STIR

int
find_num_tang_pos_for_FOV(float a, float r, int n)
{ // if r is radius, s is length of arc, a is length of chord, and n is number of azimuthal angles (num_detectors_per_ring)
  float theta = asin((a / 2.) / r); // the azimuthal angle (in radian) covered by the FOV in cylindrical geometry
  float delta = 2 * _PI / n;        // the azimuthal angle covered by a detector element in cylindrical geometry
  return int(round(2 * theta / delta));
}

float
find_length_of_arc_for_FOV(float a, float r)
{ // if r is radius, s is length of arc and a is length of chord
  // s = r * theta
  // theta =  2* asin(a/2/r)
  float theta = 2 * asin((a / 2.) / r); // the azimuthal angle (in radian) covered by the FOV in cylindrical geometry
  float s = r * theta;                  // length of arc
  return s;
}

int
main(int argc, char* argv[])
{
  if (argc < 5)
    {
      cerr << "\tUsage: " << argv[0] << " output_filename input_filename phantom_diameter single/all [axial_pos_number]\n";
      cerr
          << "\tchoose single to calculate sum of the rows of a single sinogram. axial_pos_number is required for this option.\n";
      cerr << "\tchoose all to calculate sum of the sum of the rows of all sinograms\n";
      cerr << "\tthe input projection data should have already been processed by SSRB.\n";
      exit(EXIT_FAILURE);
    }
  std::string output_filename = argv[1];
  shared_ptr<ProjData> in_proj_data_ptr = ProjData::read_from_file(argv[2]);
  const float phantom_diameter = atof(argv[3]);
  const int segment_num = 0;

  if (strcmp(argv[4], "single") == 0 && argc == 6)
    {
      const int axial_pos_num = atoi(argv[5]);
      shared_ptr<ProjDataInfo> pdi_ptr(in_proj_data_ptr->get_proj_data_info_sptr()->clone());

      if (axial_pos_num < in_proj_data_ptr->get_min_axial_pos_num(segment_num)
          || axial_pos_num > in_proj_data_ptr->get_max_axial_pos_num(segment_num))
        error("axial_pos_num is out of range!\n");

      SegmentBySinogram<float> segment_by_sino = in_proj_data_ptr->get_segment_by_sinogram(segment_num);
      Sinogram<float> sino = segment_by_sino.get_sinogram(axial_pos_num);

      // find max index in each row of sinogram and shift
      Sinogram<float> shifted_sino = in_proj_data_ptr->get_empty_sinogram(axial_pos_num, segment_num);
      for (int view = sino.get_min_view_num(); view <= sino.get_max_view_num(); ++view)
        {
          float max = 0;
          int max_idx = 0;

          // find max index in each row of sinogram
          for (int tang = sino.get_min_tangential_pos_num(); tang <= sino.get_max_tangential_pos_num(); ++tang)
            {
              if (max < sino[view][tang])
                {
                  max = sino[view][tang];
                  max_idx = tang;
                }
            }

          // Shift the bins in each row of sinogram so that the maximum stays in the centre
          // because the middle bin in each row of the sinogram corresponds to tang_pos=0
          /*
          tang_shifted = tang - (tang_max - tang_center)
          tang_center = 0
          tang_shifted = tang - tang_max
          */
          for (int tang = sino.get_min_tangential_pos_num(); tang <= sino.get_max_tangential_pos_num(); ++tang)
            {
              int shifted_tang = tang - max_idx;
              if ((shifted_tang) < sino.get_min_tangential_pos_num())
                shifted_tang = tang + sino.get_num_tangential_poss() - max_idx;
              shifted_sino[view][shifted_tang] = sino[view][tang];
            }
        }

      // sum over the shifted rows in the sinogram
      std::vector<float> squeezed_sino;
      squeezed_sino.reserve(sino.get_num_tangential_poss());
      for (int tang = sino.get_min_tangential_pos_num(); tang <= sino.get_max_tangential_pos_num(); ++tang)
        {
          float sum_bins = 0;
          for (int view = sino.get_min_view_num(); view <= sino.get_max_view_num(); ++view)
            {
              sum_bins += shifted_sino[view][tang];
            }
          squeezed_sino.push_back(sum_bins);
        }
      std::ofstream out(output_filename + ".txt");
      out << "Values of the squeezed sino for segment_num=" << segment_num << " and axial_pos_num=" << axial_pos_num
          << "along the tangential direction\n";
      std::ostream_iterator<float> output_iterator(out, "\n");
      std::copy(squeezed_sino.begin(), squeezed_sino.end(), output_iterator);
      return EXIT_SUCCESS;
    }
  if (strcmp(argv[4], "all") == 0 && argc == 5)
    {
      shared_ptr<ProjDataInfo> pdi_ptr(in_proj_data_ptr->get_proj_data_info_sptr()->clone());

      // keep sinograms out of the loop to avoid reallocations
      // initialise to something because there's no default constructor
      Sinogram<float> sino
          = in_proj_data_ptr->get_empty_sinogram(in_proj_data_ptr->get_min_axial_pos_num(segment_num), segment_num);
      Sinogram<float> shifted_sino
          = in_proj_data_ptr->get_empty_sinogram(in_proj_data_ptr->get_min_axial_pos_num(segment_num), segment_num);

      std::vector<float> squeezed_sino_all(in_proj_data_ptr->get_num_tangential_poss(), 0.0);
      for (int axial_pos_num = pdi_ptr->get_min_axial_pos_num(segment_num);
           axial_pos_num <= pdi_ptr->get_max_axial_pos_num(segment_num);
           ++axial_pos_num)
        {
          sino = in_proj_data_ptr->get_sinogram(axial_pos_num, segment_num);
          shifted_sino = in_proj_data_ptr->get_empty_sinogram(axial_pos_num, segment_num);

          // find max index in each row of sinogram and shift
          for (int view = sino.get_min_view_num(); view <= sino.get_max_view_num(); ++view)
            {
              float max = 0;
              int max_idx = 0;

              // find max index in each row of sinogram
              for (int tang = sino.get_min_tangential_pos_num(); tang <= sino.get_max_tangential_pos_num(); ++tang)
                {
                  if (max < sino[view][tang])
                    {
                      max = sino[view][tang];
                      max_idx = tang;
                    }
                }
              /*Shift the bins in each row of sinogram so that the maximum stays in the centre
                because the middle bin in each row of the sinogram corresponds to tang_pos=0
                tang_shifted = tang - (tang_max - tang_center)
                tang_center = 0
                tang_shifted = tang - tang_max
              */
              for (int tang = sino.get_min_tangential_pos_num(); tang <= sino.get_max_tangential_pos_num(); ++tang)
                {
                  int shifted_tang = tang - max_idx;
                  if ((shifted_tang) < sino.get_min_tangential_pos_num())
                    shifted_tang = tang + sino.get_num_tangential_poss() - max_idx;
                  if ((shifted_tang) > sino.get_max_tangential_pos_num())
                    shifted_tang = tang - sino.get_num_tangential_poss() - max_idx;
                  assert(shifted_tang >= sino.get_min_tangential_pos_num() && shifted_tang <= sino.get_max_tangential_pos_num());
                  shifted_sino[view][shifted_tang] = sino[view][tang];
                }
            }

          // sum over the shifted rows in sinograms
          for (int tang = sino.get_min_tangential_pos_num(); tang <= sino.get_max_tangential_pos_num(); ++tang)
            {
              float sum_bins = 0;
              for (int view = sino.get_min_view_num(); view <= sino.get_max_view_num(); ++view)
                {
                  sum_bins += shifted_sino[view][tang];
                }
              assert(tang + sino.get_max_tangential_pos_num() + 1 >= 0
                     && tang + sino.get_max_tangential_pos_num() + 1 < squeezed_sino_all.size());
              squeezed_sino_all[tang + sino.get_max_tangential_pos_num() + 1] += sum_bins;
            }
        }

      std::ofstream out(output_filename + ".txt");
      out << "Values of the squeezed sino for segment_num\n";
      std::ostream_iterator<float> output_iterator(out, "\n");
      std::copy(squeezed_sino_all.begin(), squeezed_sino_all.end(), output_iterator);

      // trim sino with 8mm band around the phantom according to NEMA NU4
      int num_tang_pos_for_FOV = find_num_tang_pos_for_FOV(phantom_diameter + 16, // 16 means 8 mm from each side of the sinogram
                                                           pdi_ptr->get_scanner_ptr()->get_inner_ring_radius(),
                                                           pdi_ptr->get_scanner_ptr()->get_num_detectors_per_ring());
      int n = sino.get_max_tangential_pos_num() - num_tang_pos_for_FOV; // total number of tang pos to trim

      squeezed_sino_all.erase(squeezed_sino_all.begin(), squeezed_sino_all.begin() + n / 2 + 1);
      squeezed_sino_all.erase(squeezed_sino_all.end() - n / 2, squeezed_sino_all.end());

      std::ofstream out_trimmed(output_filename + "_trimmed.txt");
      out_trimmed << "Values of the squeezed sino for segment_num\n";
      std::ostream_iterator<float> output_iterator_trimmed(out_trimmed, "\n");
      std::copy(squeezed_sino_all.begin(), squeezed_sino_all.end(), output_iterator_trimmed);

      // find bin value at 7 mm from center according to NEMA NU4
      float length_of_arc_for_14mm = find_length_of_arc_for_FOV(14, // 7 mm from center to both sides
                                                                pdi_ptr->get_scanner_ptr()->get_inner_ring_radius());

      int idx_mid = squeezed_sino_all.size() / 2;
      float delta_unit = length_of_arc_for_14mm / 2.2;
      // delta_unit-1/2 is number of bins from each side of central bin
      float fraction1 = delta_unit / 2 - floor(delta_unit / 2);
      float fraction2 = ceil(delta_unit / 2) - delta_unit / 2;

      assert(idx_mid - floor(delta_unit / 2) >= 0 && idx_mid - floor(delta_unit / 2) < squeezed_sino_all.size());
      assert(idx_mid + floor(delta_unit / 2) >= 0 && idx_mid + floor(delta_unit / 2) < squeezed_sino_all.size());
      assert(idx_mid - ceil(delta_unit / 2) >= 0 && idx_mid - ceil(delta_unit / 2) < squeezed_sino_all.size());
      assert(idx_mid + ceil(delta_unit / 2) >= 0 && idx_mid + ceil(delta_unit / 2) < squeezed_sino_all.size());

      float c_left = (fraction1 * squeezed_sino_all[idx_mid - floor(delta_unit / 2)]
                      + fraction2 * squeezed_sino_all[idx_mid - ceil(delta_unit / 2)]);
      float c_right = (fraction1 * squeezed_sino_all[idx_mid + floor(delta_unit / 2)]
                       + fraction2 * squeezed_sino_all[idx_mid + ceil(delta_unit / 2)]);
      float random_and_scatter_inside_14mm = (c_right + c_left) * ceil(delta_unit) / 2;

      float random_and_scatter_outside_14mm = 0;
      for (int idx = idx_mid + ceil(delta_unit / 2); idx < squeezed_sino_all.size(); ++idx)
        {
          random_and_scatter_outside_14mm += squeezed_sino_all[idx];
        }
      for (int idx = 0; idx < idx_mid - ceil(delta_unit / 2); ++idx)
        {
          random_and_scatter_outside_14mm += squeezed_sino_all[idx];
        }
      float random_and_scatter = random_and_scatter_inside_14mm + random_and_scatter_outside_14mm;

      float total_event_number = 0;
      for (int idx = 0; static_cast<unsigned>(idx) < squeezed_sino_all.size(); ++idx)
        {
          total_event_number += squeezed_sino_all[idx];
        }
      float true_event_number = total_event_number - random_and_scatter;
      std::cout << std::fixed << total_event_number << " " << random_and_scatter << " " << true_event_number << "\n";
      return EXIT_SUCCESS;
    }
  else
    {
      cerr << "\tUsage: " << argv[0] << " output_filename input_filename phantom_diameter single/all [axial_pos_number]\n";
      cerr
          << "\tchoose single to calculate sum of the rows of a single sinogram. axial_pos_number is required for this option.\n";
      cerr << "\tchoose all to calculate sum of the sum of the rows of all sinograms\n";
      cerr << "\tthe input projection data should have already been processed by SSRB.\n";
      exit(EXIT_FAILURE);
    }
}
