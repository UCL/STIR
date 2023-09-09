/*
    Copyright (C) 2023 University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details

*/

/*!
  \file
  \ingroup utilities
  \author Kris Thielemans

  \brief Perform timings
*/

#include "stir/ProjDataInterfile.h"
#include "stir/ProjDataInMemory.h"
#include "stir/DiscretisedDensity.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/IO/read_from_file.h"
#include "stir/IO/write_to_file.h"
#include "stir/recon_buildblock/ProjectorByBinPairUsingProjMatrixByBin.h"
#ifdef STIR_WITH_Parallelproj_PROJECTOR
#  include "stir/recon_buildblock/Parallelproj_projector/ProjectorByBinPairUsingParallelproj.h"
#endif
#include "stir/recon_buildblock/ProjMatrixByBinUsingRayTracing.h"
#include "stir/recon_buildblock/PoissonLogLikelihoodWithLinearModelForMeanAndProjData.h"
//#include "stir/OSMAPOSL/OSMAPOSLReconstruction.h"
#include "stir/recon_buildblock/distributable_main.h"
#include "stir/warning.h"
#include "stir/error.h"
#include "stir/Verbosity.h"
#include <iostream>
#include <cstdlib>
#include <iomanip>
#include <chrono>
#include <thread>

static void
print_usage_and_exit()
{
  std::cerr << "\nUsage:\nstir_timings [--runs num_runs]\\\n"
            << "\t[--skip-PP 1] [--skip-PMRT 1]\\\n"
            << "\t[--image image_filename]\\\n"
            << "\t--template-projdata template_proj_data_filename\n\n"
            << "Timings are reported as:\n"
            << "name\tCPU_time_in_ms\twall-clock_time_in_ms\n";
  exit(EXIT_FAILURE);
}

START_NAMESPACE_STIR

class Timings : public TimedObject
{
  typedef void (Timings::*TimedFunction)();

public:
  bool skip_PMRT;
  bool skip_PP;

  Timings(const std::string& image_filename, const std::string& template_proj_data_filename)
  {
    if (!image_filename.empty())
      this->image_sptr = read_from_file<DiscretisedDensity<3, float>>(image_filename);

    if (!template_proj_data_filename.empty())
      this->template_proj_data_sptr = ProjData::read_from_file(template_proj_data_filename);
  }

  void sleep() { std::this_thread::sleep_for(std::chrono::milliseconds(1123)); }

  void copy_image()
  {
    auto im = this->image_sptr->clone();
    delete im;
  }

  //! copy from output_proj_data_sptr to new Interfile file
  void copy_proj_data_file_to_file()
  {
    ProjDataInterfile tmp(this->template_proj_data_sptr->get_exam_info_sptr(),
                                     this->template_proj_data_sptr->get_proj_data_info_sptr(),
                                     "my_timings_copy.hs");
    tmp.fill(*this->output_proj_data_sptr);
  }

  //! copy from output_proj_data_sptr to memory object
  void copy_proj_data_file_to_mem()
  {
    ProjDataInMemory tmp(this->template_proj_data_sptr->get_exam_info_sptr(),
                         this->template_proj_data_sptr->get_proj_data_info_sptr(),
                         /* initialise*/ false);
    tmp.fill(*this->output_proj_data_sptr);
  }

  //! copy from mem_proj_data_sptr to new Interfile file
  void copy_proj_data_mem_to_file()
  {
    ProjDataInterfile tmp(this->template_proj_data_sptr->get_exam_info_sptr(),
                                     this->template_proj_data_sptr->get_proj_data_info_sptr(),
                                     "my_timings_copy.hs");
    tmp.fill(*this->mem_proj_data_sptr);
  }

  //! copy from output_proj_data_sptr to memory object
  void copy_proj_data_mem_to_mem()
  {
    ProjDataInMemory tmp(this->template_proj_data_sptr->get_exam_info_sptr(),
                         this->template_proj_data_sptr->get_proj_data_info_sptr(),
                         /* initialise*/ false);
    tmp.fill(*this->mem_proj_data_sptr);
  }
  
  void projector_setup()
  {
    this->projectors_sptr->set_up(this->template_proj_data_sptr->get_proj_data_info_sptr(), this->image_sptr);
  }

  void forward_file()
  {
    this->projectors_sptr->get_forward_projector_sptr()->forward_project(*this->output_proj_data_sptr, *this->image_sptr);
  }
  void forward_memory()
  {
    this->projectors_sptr->get_forward_projector_sptr()->forward_project(*this->mem_proj_data_sptr, *this->image_sptr);
  }
  void back_file()
  {
    this->projectors_sptr->get_back_projector_sptr()->back_project(*this->image_sptr, *this->output_proj_data_sptr);
  }
  void back_memory()
  {
    this->projectors_sptr->get_back_projector_sptr()->back_project(*this->image_sptr, *this->mem_proj_data_sptr);
  }

  void obj_func_set_up()
  {
    this->objective_function_sptr->set_up(this->image_sptr);
  }

  void obj_func_grad_no_sens()
  {
    auto im = this->image_sptr->clone();
    this->objective_function_sptr->compute_sub_gradient_without_penalty_plus_sensitivity(*im, *this->image_sptr, 0);
    delete im;
  }

  void run_it(TimedFunction f, const std::string& item, const unsigned runs = 1)
  {
    this->start_timers(true);
    for (unsigned r = runs; r != 0; --r)
      (this->*f)();
    this->stop_timers();
    std::cout << std::setw(32) << std::left << item << '\t' << std::fixed << std::setprecision(3) << std::setw(24) << std::right
              << this->get_CPU_timer_value() / runs * 1000 << '\t' << std::fixed << std::setprecision(3) << std::setw(24)
              << std::right << this->get_wall_clock_timer_value() / runs * 1000 << std::endl;
  }

  void run_all(const unsigned runs = 1)
  {
    init();
    // this->run_it(&Timings::sleep, "sleep", runs*1);
    this->run_it(&Timings::copy_image, "copy_image", runs * 20);
    this->output_proj_data_sptr->fill(1.F);
    this->run_it(&Timings::copy_proj_data_mem_to_mem, "copy_proj_data_mem_to_mem", runs * 2);
    this->run_it(&Timings::copy_proj_data_mem_to_file, "copy_proj_data_mem_to_file", runs * 2);
    this->run_it(&Timings::copy_proj_data_file_to_mem, "copy_proj_data_file_to_mem", runs * 2);
    this->run_it(&Timings::copy_proj_data_file_to_file, "copy_proj_data_file_to_file", runs * 2);
    this->objective_function_sptr.reset(new PoissonLogLikelihoodWithLinearModelForMeanAndProjData<DiscretisedDensity<3,float>>);
    this->objective_function_sptr->set_proj_data_sptr(this->mem_proj_data_sptr);
    // this->objective_function.set_num_subsets(proj_data_sptr->get_num_views()/2);
    if (!this->skip_PMRT)
      {
        this->projectors_sptr = this->pmrt_projectors_sptr;
        this->run_it(&Timings::projector_setup, "PMRT_projector_setup", runs * 10);
        this->run_it(&Timings::forward_file, "PMRT_forward_file_first", 1);
        this->run_it(&Timings::forward_file, "PMRT_forward_file", 1);
        this->run_it(&Timings::forward_memory, "PMRT_forward_memory", 1);
        this->run_it(&Timings::back_file, "PMRT_back_file_first", 1);
        this->run_it(&Timings::back_file, "PMRT_back_file", 1);
        this->run_it(&Timings::back_memory, "PMRT_back_memory", 1);
        this->objective_function_sptr->set_projector_pair_sptr(this->projectors_sptr);
        this->run_it(&Timings::obj_func_set_up, "PMRT LogLik set_up", 1);
        this->run_it(&Timings::obj_func_grad_no_sens, "PMRT LogLik grad_no_sens", 1);
        
      }
#ifdef STIR_WITH_Parallelproj_PROJECTOR
    if (!skip_PP)
      {
        this->projectors_sptr = this->parallelproj_projectors_sptr;
        this->run_it(&Timings::projector_setup, "PP_projector_setup", 1);
        this->run_it(&Timings::forward_file, "PP_forward_file_first", 1);
        this->run_it(&Timings::forward_file, "PP_forward_file", runs);
        this->run_it(&Timings::forward_memory, "PP_forward_memory", runs);
        this->run_it(&Timings::back_file, "PP_back_file_first", 1);
        this->run_it(&Timings::back_file, "PP_back_file", runs);
        this->run_it(&Timings::back_memory, "PP_back_memory", runs);
        this->objective_function_sptr->set_projector_pair_sptr(this->projectors_sptr);
        this->run_it(&Timings::obj_func_set_up, "PP LogLik set_up", 1);
        this->run_it(&Timings::obj_func_grad_no_sens, "PP LogLik grad_no_sens", 1);
      }
#endif
    // write_to_file("my_timings_backproj.hv", *this->image_sptr);
  }

  void init()
  {

    if (!this->template_proj_data_sptr)
      print_usage_and_exit();

    if (!image_sptr)
      {
        this->exam_info_sptr = this->template_proj_data_sptr->get_exam_info().create_shared_clone();
        this->image_sptr = std::make_shared<VoxelsOnCartesianGrid<float>>(
            this->exam_info_sptr, *this->template_proj_data_sptr->get_proj_data_info_sptr());
        this->image_sptr->fill(1.F);
      }
    else
      {
        this->image_sptr->fill(1.F);
        this->exam_info_sptr = this->image_sptr->get_exam_info().create_shared_clone();

        if (this->image_sptr->get_exam_info().imaging_modality.is_unknown()
            && this->template_proj_data_sptr->get_exam_info().imaging_modality.is_known())
          {
            this->exam_info_sptr->imaging_modality = this->template_proj_data_sptr->get_exam_info().imaging_modality;
          }
        else if (this->image_sptr->get_exam_info().imaging_modality
                 != this->template_proj_data_sptr->get_exam_info().imaging_modality)
          error("forward_project: Imaging modality should be the same for the image and the projection data");

        if (this->template_proj_data_sptr->get_exam_info().has_energy_information())
          {
            if (this->image_sptr->get_exam_info().has_energy_information())
              warning("Both image and template have energy information. Using the latter.");

            this->exam_info_sptr->set_energy_information_from(this->template_proj_data_sptr->get_exam_info());
          }
      }

    // projection data set-up
    {
      std::string output_filename = "my_timings.hs";
      this->output_proj_data_sptr
        = std::make_shared<ProjDataInterfile>(this->exam_info_sptr, this->template_proj_data_sptr->get_proj_data_info_sptr(),
                                              output_filename, std::ios::in | std::ios::out | std::ios::trunc);
      this->mem_proj_data_sptr
        = std::make_shared<ProjDataInMemory>(this->exam_info_sptr, this->template_proj_data_sptr->get_proj_data_info_sptr());
    }

    // projector set-up
    {
      auto PM_sptr = std::make_shared<ProjMatrixByBinUsingRayTracing>();
      PM_sptr->set_num_tangential_LORs(5);
      this->pmrt_projectors_sptr = std::make_shared<ProjectorByBinPairUsingProjMatrixByBin>(PM_sptr);

#ifdef STIR_WITH_Parallelproj_PROJECTOR
      this->parallelproj_projectors_sptr = std::make_shared<ProjectorByBinPairUsingParallelproj>();
#endif
    }
  }

  // protected:
  shared_ptr<DiscretisedDensity<3, float>> image_sptr;
  shared_ptr<ProjData> output_proj_data_sptr;
  shared_ptr<ProjDataInMemory> mem_proj_data_sptr;
  shared_ptr<ProjectorByBinPair> projectors_sptr;
  shared_ptr<ProjectorByBinPairUsingProjMatrixByBin> pmrt_projectors_sptr;
#ifdef STIR_WITH_Parallelproj_PROJECTOR
  shared_ptr<ProjectorByBinPairUsingParallelproj> parallelproj_projectors_sptr;
#endif
  shared_ptr<ProjData> template_proj_data_sptr;
  shared_ptr<ExamInfo> exam_info_sptr;
  shared_ptr<PoissonLogLikelihoodWithLinearModelForMeanAndProjData<DiscretisedDensity<3,float>>> objective_function_sptr;
};

END_NAMESPACE_STIR

#ifdef STIR_MPI
int stir::distributable_main(int argc, char **argv)
#else
int main(int argc, char **argv)
#endif
{
  using namespace stir;
  Verbosity::set(0);

  std::string image_filename;
  std::string template_proj_data_filename;
  std::string prog_name = argv[0];
  unsigned num_runs = 3;
  bool skip_PMRT = false;
  bool skip_PP = false;

  ++argv;
  --argc;
  while (argc > 1)
    {
      if (!strcmp(argv[0], "--image"))
        image_filename = argv[1];
      else if (!strcmp(argv[0], "--template-projdata"))
        template_proj_data_filename = argv[1];
      else if (!strcmp(argv[0], "--runs"))
        num_runs = std::atoi(argv[1]);
      else if (!strcmp(argv[0], "--skip-PMRT"))
        skip_PMRT = std::atoi(argv[1]) != 0;
      else if (!strcmp(argv[0], "--skip-PP"))
        skip_PP = std::atoi(argv[1]) != 0;
      else
        print_usage_and_exit();
      argv += 2;
      argc -= 2;
    }

  if (argc > 0)
    print_usage_and_exit();

  Timings timings(image_filename, template_proj_data_filename);
  timings.skip_PMRT = skip_PMRT;
  timings.skip_PP = skip_PP;

  timings.run_all(num_runs);
  return EXIT_SUCCESS;
}
