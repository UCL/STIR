/*
  A minimal utility to profile CPU and CUDA relative difference priors in STIR.
  Author: Modified from stir_timings by ChatGPT for profiling priors only.
*/

#include "stir/DiscretisedDensity.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/IO/read_from_file.h"
#include "stir/Succeeded.h"
#include "stir/warning.h"
#include "stir/error.h"
#include "stir/Verbosity.h"
#include "stir/num_threads.h"
#include "/usr/local/cuda/include/cuda_profiler_api.h"
#include <nvToolsExt.h>
#include "stir/cuda_utilities.h"
#include <cuda_runtime.h>
#include "stir/recon_buildblock/QuadraticPrior.h"
#include "stir/recon_buildblock/GibbsQuadraticPrior.h"

#ifdef STIR_WITH_CUDA
#include "stir/recon_buildblock/CUDA/CudaRelativeDifferencePrior.h"
#endif

#include <iostream>
#include <string>
#include <chrono>
#include <ctime>
#include <cstdlib>
#include <cstring>

START_NAMESPACE_STIR

using stir::DiscretisedDensity;
using stir::VoxelsOnCartesianGrid;
using stir::RelativeDifferencePrior;
using stir::QuadraticPrior;
using stir::GibbsQuadraticPrior;

void print_usage_and_exit()
{
  std::cerr << "\nUsage:\nstir_prior_timings --image image.hv [--runs N] [--threads T] [--name label]\n";
  std::cerr << "This program times CPU and CUDA relative difference prior computations.\n";
  std::exit(EXIT_FAILURE);
}

END_NAMESPACE_STIR

int main(int argc, char** argv)
{
  using namespace stir;
  Verbosity::set(0);

  std::string image_filename;
  std::string name;
  int runs = 1;

  // --- Parse command-line arguments ---
  ++argv; --argc;
  while (argc > 1)
  {
    if (!strcmp(argv[0], "--image"))
      image_filename = argv[1];
    else if (!strcmp(argv[0], "--runs"))
      runs = std::atoi(argv[1]);
    // else if (!strcmp(argv[0], "--threads"))
    //   threads = std::atoi(argv[1]);
    else if (!strcmp(argv[0], "--name"))
      name = argv[1];
    else
      print_usage_and_exit();
    argv += 2;
    argc -= 2;
  }

  if (argc > 0 || image_filename.empty())
    print_usage_and_exit();

  // set_num_threads(threads);
  // std::cerr << "Using " << threads << " threads.\n";

  // --- Load data ---

  shared_ptr<VoxelsOnCartesianGrid<float>> image_sptr;
  image_sptr = read_from_file<VoxelsOnCartesianGrid<float>>(image_filename);
  auto quadratic_prior = std::make_shared<stir::QuadraticPrior<float>>(false, 1.F);
  auto gibbs_quadratic_prior = std::make_shared<stir::GibbsQuadraticPrior<float>>(false, 1.F);

  quadratic_prior->set_up(image_sptr);
  gibbs_quadratic_prior->set_up(image_sptr);
  // auto im = image_sptr->clone(); // 'im' is not needed for compute_value(*image_sptr)
  nvtxRangePushA("compute_value");

  double val       = quadratic_prior->compute_value(*image_sptr); 
  double gibbs_val = gibbs_quadratic_prior->compute_value(*image_sptr); 
  std::cout<< "Value_quad_prior: " << val << std::endl;
  std::cout<< "Gibbs_Value_quad_prior: " << gibbs_val << std::endl;
  
  nvtxRangePop();
  

// #ifdef STIR_WITH_CUDA
//   // --- CUDA RelativeDifferencePrior ---
//   {
//     // auto quadratic_prior = std::make_shared<QuadraticPrior<float>>(false, 1.F);
//     auto cuda_prior = std::make_shared<CudaRelativeDifferencePrior<float>>(false, 1.F, 2.F, 0.1F);
//     cuda_prior->set_up(image_sptr);
//     // quadratic_prior->set_up(image_sptr);
//     auto im = image_sptr->clone();

//     for (int i = 0; i < 5; ++i)
//       cuda_prior->compute_gradient(*im, *image_sptr);
//     cudaDeviceSynchronize();

//     cudaProfilerStart();
//     nvtxRangePushA("compute_gradient");
//     for (int i = 0; i < runs; ++i) {
//       cuda_prior->compute_gradient(*im, *image_sptr);
//       cudaDeviceSynchronize();
//     }
//     nvtxRangePop();

//     nvtxRangePushA("compute_value");
//     for (int i = 0; i < runs; ++i) {
//       (void)cuda_prior->compute_value(*image_sptr);
//       cudaDeviceSynchronize();
//     }
//     nvtxRangePop();
//     cudaProfilerStop();

//     delete im;
//   }
// #endif

  return EXIT_SUCCESS;
}
