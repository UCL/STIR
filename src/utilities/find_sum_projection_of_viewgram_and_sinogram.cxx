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

  \brief This program reads a projection data file and squeezes a given sinogrm or viewgram of it (from a 2D matrix to a 1D vector).

  \author Parisa Khateri

*/

#include "stir/ProjData.h"
#include "stir/IO/interfile.h"
#include "stir/utilities.h"
#include "stir/Bin.h"

#include <fstream>
#include <iostream>
#include "stir/ProjDataFromStream.h"
#include "stir/Sinogram.h"
#include "stir/IO/read_from_file.h"
#include "stir/SegmentByView.h"
#include "stir/ProjDataInterfile.h"
#include "stir/ProjDataInfo.h"




#ifndef STIR_NO_NAMESPACES
using std::cerr;
#endif

USING_NAMESPACE_STIR

int main(int argc, char *argv[])
{
    if(argc<6)
    {
      cerr<<"\tUsage: " << argv[0] << " output_filename input_filename segment_number viewgram/simogram view_number/axial_pos_number\n";
      cerr<<"\tviewgram: to calculate sum projection for a viewgram\n";
      cerr<<"\tsinogram: to calculate sum projection for a sinogram\n";
      exit(EXIT_FAILURE);
    }
    std::string output_filename=argv[1];
    shared_ptr<ProjData>  in_proj_data_ptr = ProjData::read_from_file(argv[2]);
    if (strcmp(argv[4], "viewgram")==0)
    {
      const int segment_num = atoi(argv[3]);
      const int view_num = atoi(argv[5]);
      shared_ptr<ProjDataInfo> pdi_ptr (in_proj_data_ptr->get_proj_data_info_sptr()->clone());
      std::cout<<"[min_tang_pos, max_tang_pos]=["<<pdi_ptr->get_min_view_num()<<", "<<pdi_ptr->get_max_view_num()<<"]\n";
      if (pdi_ptr->get_num_tangential_poss()/2.==0)
        std::cout<<"num_tang_pos is even\n";

      if (segment_num <in_proj_data_ptr->get_min_segment_num() || segment_num > in_proj_data_ptr->get_max_segment_num())
        error("segment_num is out of range!\n");
      if (view_num <in_proj_data_ptr->get_min_view_num() || view_num > in_proj_data_ptr->get_max_view_num())
        error("view_num is out of range!\n");

      SegmentByView<float> segment_by_view = in_proj_data_ptr->get_segment_by_view(segment_num);
      Viewgram<float> view = segment_by_view.get_viewgram(view_num);
      std::vector<float> squeezed_view;
      squeezed_view.reserve(view.get_num_tangential_poss());
      for (int tang = view.get_min_tangential_pos_num();
               tang <= view.get_max_tangential_pos_num();
               ++tang)
      {
        float sum_bins = 0;
        for (int ax = view.get_min_axial_pos_num();
                 ax <= view.get_max_axial_pos_num();
                 ++ax)
        {
          sum_bins+= view[ax][tang];
        }
        squeezed_view.push_back(sum_bins);
      }
      std::ofstream out(output_filename+".txt");
      out<<"Values of the squeezed view for segment_num="<<segment_num<<" and view_num="<<view_num<<"along the tangential direction\n";
      std::ostream_iterator<float> output_iterator(out, "\n");
      std::copy(squeezed_view.begin(), squeezed_view.end(), output_iterator);
      return EXIT_SUCCESS;
    }
    else if (strcmp(argv[4], "sinogram")==0)
    {
      const int segment_num = atoi(argv[3]);
      const int axial_pos_num = atoi(argv[5]);
      shared_ptr<ProjDataInfo> pdi_ptr (in_proj_data_ptr->get_proj_data_info_sptr()->clone());
      std::cout<<"[min_axial_pos, max_axial_pos]=["<<pdi_ptr->get_min_axial_pos_num(segment_num)<<", "<<pdi_ptr->get_max_axial_pos_num(segment_num)<<"]\n";

      if (segment_num <in_proj_data_ptr->get_min_segment_num() || segment_num > in_proj_data_ptr->get_max_segment_num())
        error("segment_num is out of range!\n");
      if (axial_pos_num <in_proj_data_ptr->get_min_axial_pos_num(segment_num) || axial_pos_num > in_proj_data_ptr->get_max_axial_pos_num(segment_num))
        error("axial_pos_num is out of range!\n");

      SegmentBySinogram<float> segment_by_sino = in_proj_data_ptr->get_segment_by_sinogram(segment_num);
      Sinogram<float> sino = segment_by_sino.get_sinogram(axial_pos_num);
      std::vector<float> squeezed_sino;
      squeezed_sino.reserve(sino.get_num_tangential_poss());
      for (int tang = sino.get_min_tangential_pos_num();
               tang <= sino.get_max_tangential_pos_num();
               ++tang)
      {
        float sum_bins = 0;
        for (int view = sino.get_min_view_num();
                 view <= sino.get_max_view_num();
                 ++view)
        {
          sum_bins+= sino[view][tang];
        }
        squeezed_sino.push_back(sum_bins);
      }
      std::ofstream out(output_filename+".txt");
      out<<"Values of the squeezed sino for segment_num="<<segment_num<<" and axial_pos_num="<<axial_pos_num<<"along the tangential direction\n";
      std::ostream_iterator<float> output_iterator(out, "\n");
      std::copy(squeezed_sino.begin(), squeezed_sino.end(), output_iterator);
      return EXIT_SUCCESS;
    }
    else
    {
      cerr<<"\tYou should determine either 'sinogram' or 'viewgram'\n";
      cerr<<"\tUsage: " << argv[0] << " output_filename input_filename segment_number viewgram/simogram view_number/axial_pos_number\n";
      cerr<<"\tviewgram: to calculate sum projection for a viewgram\n";
      cerr<<"\tsinogram: to calculate sum projection for a sinogram\n";
      exit(EXIT_FAILURE);
    }
}
