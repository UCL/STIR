/*
    Copyright (C) 2023, 2024 University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details

*/

/*!.
  \file
  \ingroup utilities
  \author Kris Thielemans

  \brief Perform timings

  This utility performs timings of various operations. This is mostly useful for developers,
  but you could use it to optimise the number of OpenMP threads to use for your data.

  Run the utility without any arguments to get a help message.
  If you want to know what is actually timed, you will have to look at the source code.
*/

#include "stir/KeyParser.h"
#include "stir/ProjDataInterfile.h"
#include "stir/ProjDataInMemory.h"
#include "stir/DiscretisedDensity.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/IO/read_from_file.h"
#include "stir/IO/write_to_file.h"
#ifndef MINI_STIR
#  include "stir/recon_buildblock/ProjectorByBinPairUsingProjMatrixByBin.h"
#endif
#ifdef STIR_WITH_Parallelproj_PROJECTOR
#  include "stir/recon_buildblock/Parallelproj_projector/ProjectorByBinPairUsingParallelproj.h"
#endif
#ifndef MINI_STIR
#  include "stir/recon_buildblock/ProjMatrixByBinUsingRayTracing.h"
#  include "stir/recon_buildblock/PoissonLogLikelihoodWithLinearModelForMeanAndProjData.h"
#  include "stir/recon_buildblock/RelativeDifferencePrior.h"
#  ifdef STIR_WITH_CUDA
#    include "stir/recon_buildblock/CUDA/CudaRelativeDifferencePrior.h"
#  endif
//#include "stir/OSMAPOSL/OSMAPOSLReconstruction.h"
#endif
#include "stir/recon_buildblock/distributable_main.h"
#include "stir/warning.h"
#include "stir/error.h"
#include "stir/num_threads.h"
#include "stir/Verbosity.h"
#include <iostream>
#include <cstdlib>
#include <iomanip>
#include <chrono>
#include <thread>

static void
print_usage_and_exit()
{
  std::cerr << "\nUsage:\nstir_timings [--name some_string] [--threads num_threads] [--runs num_runs]\\\n"
            << "\t[--skip-BB 1] [--skip-PP 1] [--skip-PMRT 1] [--skip-priors 1]\\\n"
            << "\t[--projector_par_filename parfile]\\\n"
            << "\t[--image image_filename]\\\n"
            << "\t--template-projdata template_proj_data_filename\n\n"
            << "skip BB: basic building blocks; PP: Parallelproj; PMRT: ray-tracing matrix; priors: prior timing\n\n"
            << "Timings are reported to stdout as:\n"
            << "name\ttiming_name\tCPU_time_in_ms\twall-clock_time_in_ms\n";
  std::cerr << "\nExample projector-pair par-file (the following corresponds to the PMRT configuration normally used)\n"
            << "projector pair parameters:=\n"
            << "   type := Matrix\n"
            << "   Projector Pair Using Matrix Parameters :=\n"
            << "     Matrix type := Ray Tracing\n"
            << "     Ray tracing matrix parameters :=\n"
            << "       number of rays in tangential direction to trace for each bin:= 5\n"
            << "       disable caching := 0\n"
            << "     End Ray tracing matrix parameters :=\n"
            << "   End Projector Pair Using Matrix Parameters :=\n"
            << "End:=\n";
  std::exit(EXIT_FAILURE);
}

START_NAMESPACE_STIR

class Timings : public TimedObject
{
  typedef void (Timings::*TimedFunction)();

public:
  //! Use as prefix for all output
  std::string name;
  // variables that select timings
  bool skip_BB;     //! skip basic building blocks
  bool skip_PMRT;   //! skip ProjMatrixByBinUsingRayTracing
  bool skip_PP;     //! skip Parallelproj
  bool skip_priors; //! skip GeneralisedPrior
  // variables used for running timings
  shared_ptr<VoxelsOnCartesianGrid<float>> image_sptr;
  shared_ptr<ProjData> output_proj_data_sptr;
  shared_ptr<ProjDataInMemory> mem_proj_data_sptr;
  shared_ptr<ProjDataInMemory> mem_proj_data_sptr2;
  std::vector<float> v1;
  std::vector<float> v2;
  shared_ptr<ProjectorByBinPair> projectors_sptr;
#ifndef MINI_STIR
  shared_ptr<ProjectorByBinPairUsingProjMatrixByBin> pmrt_projectors_sptr;
#endif
#ifdef STIR_WITH_Parallelproj_PROJECTOR
  shared_ptr<ProjectorByBinPairUsingParallelproj> parallelproj_projectors_sptr;
#endif
  shared_ptr<ProjectorByBinPair> parsed_projectors_sptr;
  shared_ptr<ProjData> template_proj_data_sptr;
  shared_ptr<ExamInfo> exam_info_sptr;
#ifndef MINI_STIR
  shared_ptr<PoissonLogLikelihoodWithLinearModelForMeanAndProjData<DiscretisedDensity<3, float>>> objective_function_sptr;

  shared_ptr<GeneralisedPrior<DiscretisedDensity<3, float>>> prior_sptr;
#endif
  // basic methods
  Timings(const std::string& image_filename, const std::string& template_proj_data_filename)
  {
    if (!image_filename.empty())
      this->image_sptr = read_from_file<VoxelsOnCartesianGrid<float>>(image_filename);

    if (!template_proj_data_filename.empty())
      this->template_proj_data_sptr = ProjData::read_from_file(template_proj_data_filename);
  }

  void run_it(TimedFunction f, const std::string& item, const unsigned runs = 1);
  void run_projectors(const std::string& prefix, const shared_ptr<ProjectorByBinPair> proj_sptr, const unsigned runs);
  void run_all(const unsigned runs = 1);
  void init();

  // functions that are timed

  //! Test function that could be used to see if reported timings are correct
  /*! CPU time should be close to zero, wall-clock time close to 1123ms */
  void sleep()
  {
    std::this_thread::sleep_for(std::chrono::milliseconds(1123));
  }

  template <class T>
  static void copy_add(T& t)
  {
    T c(t);
    c += t;
  }

  template <class T>
  static void copy_mult(T& t)
  {
    T c(t);
    c *= t;
  }

  void copy_image()
  {
    auto im = this->image_sptr->clone();
    delete im;
  }

  void copy_add_image()
  {
    copy_add(*this->image_sptr);
  }

  void copy_mult_image()
  {
    copy_mult(*this->image_sptr);
  }

  void copy_std_vector()
  {
    std::copy(this->v1.begin(), this->v1.end(), this->v2.begin());
  }
  void create_std_vector()
  {
    std::vector<float> tmp(this->v1.size());
    tmp[0] = 1; // assign something to avoid compiler warnings of unused variable
  }
  //! create proj_data in memory object
  void create_proj_data_in_mem_no_init()
  {
    ProjDataInMemory tmp(this->template_proj_data_sptr->get_exam_info_sptr(),
                         this->template_proj_data_sptr->get_proj_data_info_sptr(),
                         /* initialise*/ false);
  }
  void create_proj_data_in_mem_init()
  {
    ProjDataInMemory tmp(this->template_proj_data_sptr->get_exam_info_sptr(),
                         this->template_proj_data_sptr->get_proj_data_info_sptr(),
                         /* initialise*/ true);
  }
  //! call ProjDataInMemory::fill(ProjDataInMemory&)
  void copy_only_proj_data_mem_to_mem()
  {
    this->mem_proj_data_sptr2->fill(*this->mem_proj_data_sptr);
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

  void copy_add_proj_data_mem()
  {
    copy_add(*this->mem_proj_data_sptr);
  }

  void copy_mult_proj_data_mem()
  {
    copy_mult(*this->mem_proj_data_sptr);
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

#ifndef MINI_STIR
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

  void prior_grad()
  {
    auto im = this->image_sptr->clone();
    this->prior_sptr->compute_gradient(*im, *this->image_sptr);
    delete im;
  }

  void prior_value()
  {
    auto im = this->image_sptr->clone();
    auto v = this->prior_sptr->compute_value(*this->image_sptr);
    v += 2; // to avoid compiler warning about unused variable
    delete im;
  }
#endif
};

void
Timings::run_it(TimedFunction f, const std::string& item, const unsigned runs)
{
  this->start_timers(true);
  for (unsigned r = runs; r != 0; --r)
    (this->*f)();
  this->stop_timers();
  std::cout << name << '\t' << std::setw(32) << std::left << item << '\t' << std::fixed << std::setprecision(3) << std::setw(24)
            << std::right << this->get_CPU_timer_value() / runs * 1000 << '\t' << std::fixed << std::setprecision(3)
            << std::setw(24) << std::right << this->get_wall_clock_timer_value() / runs * 1000 << std::endl;
}

void
Timings::run_projectors(const std::string& prefix, const shared_ptr<ProjectorByBinPair> proj_sptr, const unsigned runs)
{
  this->projectors_sptr = proj_sptr;
  this->run_it(&Timings::projector_setup, prefix + "_projector_setup", 1);
  this->run_it(&Timings::forward_file, prefix + "_forward_file_first", 1);
  this->run_it(&Timings::forward_file, prefix + "_forward_file", runs);
  this->run_it(&Timings::forward_memory, prefix + "_forward_memory", runs);
  this->run_it(&Timings::back_file, prefix + "_back_file_first", 1);
  this->run_it(&Timings::back_file, prefix + "_back_file", runs);
  this->run_it(&Timings::back_memory, prefix + "_back_memory", runs);
#ifndef MINI_STIR
  this->objective_function_sptr->set_projector_pair_sptr(this->projectors_sptr);
  this->run_it(&Timings::obj_func_set_up, prefix + "_LogLik set_up", 1);
  this->run_it(&Timings::obj_func_grad_no_sens, prefix + "_LogLik grad_no_sens", 1);
#endif
}
void
Timings::run_all(const unsigned runs)
{
  this->init();
  // this->run_it(&Timings::sleep, "sleep", runs*1);
  this->output_proj_data_sptr->fill(1.F);
  if (!this->skip_BB)
    {
      this->mem_proj_data_sptr2
          = std::make_shared<ProjDataInMemory>(this->exam_info_sptr, this->template_proj_data_sptr->get_proj_data_info_sptr());
      this->v1.resize(this->template_proj_data_sptr->size_all());
      this->v2.resize(this->template_proj_data_sptr->size_all());
      this->run_it(&Timings::copy_image, "copy_image", runs * 20);
      this->run_it(&Timings::copy_add_image, "copy_add_image", runs * 20);
      this->run_it(&Timings::copy_mult_image, "copy_mult_image", runs * 20);
      // reference timings: std::vector should be fast
      this->run_it(&Timings::create_std_vector, "create_vector_of_size_projdata", runs * 2);
      this->run_it(&Timings::copy_std_vector, "copy_std_vector_of_size_projdata", runs * 2);
      v1.clear();
      v2.clear();
      this->run_it(&Timings::create_proj_data_in_mem_no_init, "create_proj_data_in_mem_no_init", runs * 2);
      this->run_it(&Timings::create_proj_data_in_mem_init, "create_proj_data_in_mem_init", runs * 2);
      this->run_it(&Timings::copy_only_proj_data_mem_to_mem, "copy_proj_data_mem_to_mem", runs * 2);
      this->run_it(&Timings::copy_proj_data_mem_to_mem, "create_copy_proj_data_mem_to_mem", runs * 2);
      this->mem_proj_data_sptr2.reset(); // no longer used
      this->run_it(&Timings::copy_proj_data_mem_to_file, "create_copy_proj_data_mem_to_file", runs * 2);
      this->run_it(&Timings::copy_proj_data_file_to_mem, "create_copy_proj_data_file_to_mem", runs * 2);
      this->run_it(&Timings::copy_proj_data_file_to_file, "create_copy_proj_data_file_to_file", runs * 2);
      this->run_it(&Timings::copy_add_proj_data_mem, "copy_add_proj_data_mem", runs * 2);
      this->run_it(&Timings::copy_mult_proj_data_mem, "copy_mult_proj_data_mem", runs * 2);
    }
#ifndef MINI_STIR
  this->objective_function_sptr.reset(new PoissonLogLikelihoodWithLinearModelForMeanAndProjData<DiscretisedDensity<3, float>>);
  this->objective_function_sptr->set_proj_data_sptr(this->mem_proj_data_sptr);
  // this->objective_function.set_num_subsets(proj_data_sptr->get_num_views()/2);
  if (!this->skip_PMRT)
    {
      this->run_projectors("PMRT", this->pmrt_projectors_sptr, 1);
    }
#endif
#ifdef STIR_WITH_Parallelproj_PROJECTOR
  if (!skip_PP)
    {
      this->run_projectors("PP", this->parallelproj_projectors_sptr, runs);
    }
#endif
  if (parsed_projectors_sptr)
    {
      this->run_projectors("parsed", this->parsed_projectors_sptr, runs);
    }
    // write_to_file("my_timings_backproj.hv", *this->image_sptr);

#ifndef MINI_STIR
  if (!skip_priors)
    {
      {
        this->prior_sptr = std::make_shared<RelativeDifferencePrior<float>>(false, 1.F, 2.F, 0.1F);
        this->prior_sptr->set_up(this->image_sptr);
        this->run_it(&Timings::prior_value, "RDP_value", runs * 10);
        this->run_it(&Timings::prior_grad, "RDP_grad", runs * 10);
        this->prior_sptr = nullptr;
      }
#  ifdef STIR_WITH_CUDA
      {
        this->prior_sptr = std::make_shared<CudaRelativeDifferencePrior<float>>(false, 1.F, 2.F, 0.1F);
        this->prior_sptr->set_up(this->image_sptr);
        this->run_it(&Timings::prior_value, "Cuda_RDP_value", runs * 30);
        this->run_it(&Timings::prior_grad, "Cuda_RDP_grad", runs * 30);
        this->prior_sptr = nullptr;
      }
#  endif
    }
#endif
}

void
Timings::init()
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
    this->output_proj_data_sptr = std::make_shared<ProjDataInterfile>(this->exam_info_sptr,
                                                                      this->template_proj_data_sptr->get_proj_data_info_sptr(),
                                                                      output_filename,
                                                                      std::ios::in | std::ios::out | std::ios::trunc);
    this->mem_proj_data_sptr
        = std::make_shared<ProjDataInMemory>(this->exam_info_sptr, this->template_proj_data_sptr->get_proj_data_info_sptr());
  }

  // projector set-up
  {
#ifndef MINI_STIR
    auto PM_sptr = std::make_shared<ProjMatrixByBinUsingRayTracing>();
    PM_sptr->set_num_tangential_LORs(5);
    this->pmrt_projectors_sptr = std::make_shared<ProjectorByBinPairUsingProjMatrixByBin>(PM_sptr);
#endif
#ifdef STIR_WITH_Parallelproj_PROJECTOR
    this->parallelproj_projectors_sptr = std::make_shared<ProjectorByBinPairUsingParallelproj>();
#endif
  }
}

END_NAMESPACE_STIR

#ifdef STIR_MPI
int
stir::distributable_main(int argc, char** argv)
#else
int
main(int argc, char** argv)
#endif
{
  using namespace stir;
  Verbosity::set(0);

  std::string image_filename;
  std::string template_proj_data_filename;
  std::string projector_par_filename;
  std::string prog_name = argv[0];
  unsigned num_runs = 3;
  int num_threads = get_default_num_threads();
  bool skip_BB = false;
  bool skip_PMRT = false;
  bool skip_PP = false;
  bool skip_priors = false;
  // prefix output with this string
  std::string name;

  ++argv;
  --argc;
  while (argc > 1)
    {
      if (!strcmp(argv[0], "--name"))
        name = argv[1];
      else if (!strcmp(argv[0], "--image"))
        image_filename = argv[1];
      else if (!strcmp(argv[0], "--template-projdata"))
        template_proj_data_filename = argv[1];
      else if (!strcmp(argv[0], "--runs"))
        num_runs = std::atoi(argv[1]);
      else if (!strcmp(argv[0], "--threads"))
        num_threads = std::atoi(argv[1]);
      else if (!strcmp(argv[0], "--skip-BB"))
        skip_BB = std::atoi(argv[1]) != 0;
      else if (!strcmp(argv[0], "--skip-PMRT"))
        skip_PMRT = std::atoi(argv[1]) != 0;
      else if (!strcmp(argv[0], "--skip-PP"))
        skip_PP = std::atoi(argv[1]) != 0;
      else if (!strcmp(argv[0], "--skip-priors"))
        skip_priors = std::atoi(argv[1]) != 0;
      else if (!strcmp(argv[0], "--projector_par_filename"))
        projector_par_filename = argv[1];
      else
        print_usage_and_exit();
      argv += 2;
      argc -= 2;
    }

  if (argc > 0)
    print_usage_and_exit();

  set_num_threads(num_threads);
  std::cerr << "Using " << num_threads << " threads.\n";

  Timings timings(image_filename, template_proj_data_filename);
  timings.name = name;
  timings.skip_BB = skip_BB;
  timings.skip_PMRT = skip_PMRT;
  timings.skip_PP = skip_PP;
  timings.skip_priors = skip_priors;
  if (!projector_par_filename.empty())
    {
      KeyParser parser;
      parser.add_start_key("Projector pair parameters");
      parser.add_parsing_key("type", &timings.parsed_projectors_sptr);
      parser.add_stop_key("END");
      if (!parser.parse(projector_par_filename.c_str()))
        error("Error parsing " + projector_par_filename);
    }

  timings.run_all(num_runs);
  return EXIT_SUCCESS;
}
