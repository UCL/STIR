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
#include <cuda_runtime.h>
#include "stir/recon_buildblock/GibbsQuadraticPrior.h"
#include "stir/recon_buildblock/GibbsRelativeDifferencePrior.h"
#include "stir/recon_buildblock/QuadraticPrior.h"
#include "stir/recon_buildblock/RelativeDifferencePrior.h"
#include "stir/BasicCoordinate.h"

#ifdef STIR_WITH_CUDA
#include "stir/recon_buildblock/CUDA/CudaRelativeDifferencePrior.h"
#endif

#include <iostream>
#include <string>
#include <chrono>
#include <ctime>
#include <cstdlib>
#include <cstring>
#include <functional>

int runs = 1;

// Utility to time a callable using CUDA events, with warm-up runs
float Time_it(const std::function<void()>& func, int warmup_runs = 5) {
  cudaEvent_t start, stop;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);

  // Warm-up runs (not timed)
  for (int i = 0; i < warmup_runs; ++i) {
    func();
  }

  cudaEventRecord(start, 0);
  for (int i = 0; i < runs; ++i) {
    func();
  }
  cudaEventRecord(stop, 0);
  cudaEventSynchronize(stop);
  float milliseconds = 0;
  cudaEventElapsedTime(&milliseconds, start, stop);

  cudaEventDestroy(start);
  cudaEventDestroy(stop);
  return milliseconds / runs;
}



START_NAMESPACE_STIR

using stir::DiscretisedDensity;
using stir::VoxelsOnCartesianGrid;
using stir::RelativeDifferencePrior;
using stir::GibbsQuadraticPrior;
using stir::QuadraticPrior;
using stir::BasicCoordinate;

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
  std::string kappa_filename;

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
    else if (!strcmp(argv[0], "--kappa"))
      kappa_filename = argv[1];
    else
      print_usage_and_exit();
    argv += 2;
    argc -= 2;
  }

  if (argc > 0 || image_filename.empty())
    print_usage_and_exit();

  shared_ptr<VoxelsOnCartesianGrid<float>> image_sptr;
  shared_ptr<VoxelsOnCartesianGrid<float>> kappa_sptr;
  image_sptr = read_from_file<VoxelsOnCartesianGrid<float>>(image_filename);
  if (!kappa_filename.empty())
    kappa_sptr = read_from_file<VoxelsOnCartesianGrid<float>>(kappa_filename);

  // PARSING TEST
  // auto gibbs_quad_prior_sptr = std::make_shared<CudaGibbsQuadraticPrior<float>>();
  // //auto quad_prior_sptr = std::make_shared<QuadraticPrior<float>>();
  // auto parsing_file = "/root/devel/buildConda/sources/STIR/src/utilities/test_parsing.par";
  // gibbs_quad_prior_sptr->parse(parsing_file);
  // gibbs_quad_prior_sptr->set_penalisation_factor(1.0f);
  // gibbs_quad_prior_sptr->set_up(image_sptr);
  // std::cout << "value = "<< gibbs_quad_prior_sptr->compute_value(*image_sptr) << std::endl;

  // quad_prior_sptr->parse(parsing_file);
  // quad_prior_sptr->set_penalisation_factor(1.0f);
  // quad_prior_sptr->set_up(image_sptr);
  // std::cout << "value = "<< quad_prior_sptr->compute_value(*image_sptr) << std::endl;

  //VALUE TEST
  double val = 0;

  auto gibbs_prior_sptr = std::make_shared<GibbsQuadraticPrior<float>>();
  gibbs_prior_sptr->set_penalisation_factor(1.F);
  gibbs_prior_sptr->set_up(image_sptr);
  if(kappa_sptr) gibbs_prior_sptr->set_kappa_sptr(kappa_sptr);
  val =  gibbs_prior_sptr->compute_value(*image_sptr);
  std::cout << "gibbs value =  " << val << std::endl;

  auto CUDA_gibbs_prior_sptr = std::make_shared<CudaGibbsQuadraticPrior<float>>();
  CUDA_gibbs_prior_sptr->set_penalisation_factor(1.F);
  CUDA_gibbs_prior_sptr->set_up(image_sptr);
  if(kappa_sptr) CUDA_gibbs_prior_sptr->set_kappa_sptr(kappa_sptr);
  val = CUDA_gibbs_prior_sptr->compute_value(*image_sptr);
  std::cout << "CUDA gibbs value =  " << val << std::endl;

  auto original_prior_sptr = std::make_shared<QuadraticPrior<float>>(false, 1.F);
  if(kappa_sptr) original_prior_sptr->set_kappa_sptr(kappa_sptr);
  original_prior_sptr->set_up(image_sptr);
  val = original_prior_sptr->compute_value(*image_sptr);
  std::cout << "original value =  " << val << std::endl;

  //gradient TEST 
  std::cout<<std::endl;
  shared_ptr<VoxelsOnCartesianGrid<float>> output_sptr,input_sptr;
  output_sptr.reset(image_sptr->clone());
  input_sptr.reset(image_sptr->clone());

  val=0;
  gibbs_prior_sptr->compute_gradient(*output_sptr, *image_sptr);
  for (int z = output_sptr->get_min_index(); z <= output_sptr->get_max_index(); ++z)
  for (int y = output_sptr->operator[](z).get_min_index(); y <= output_sptr->operator[](z).get_max_index(); ++y)
  for (int x = output_sptr->operator[](z)[y].get_min_index(); x <= output_sptr->operator[](z)[y].get_max_index(); ++x)
  {
    val += (*output_sptr)[z][y][x] * (*input_sptr)[z][y][x];
  }
  std::cout << "gibbs gradient dot input = " << val <<"("<<gibbs_prior_sptr->compute_gradient_times_input(*input_sptr,*image_sptr)<<")"<< std::endl;

  val=0;
  CUDA_gibbs_prior_sptr->compute_gradient(*output_sptr, *image_sptr);
  for (int z = output_sptr->get_min_index(); z <= output_sptr->get_max_index(); ++z)
  for (int y = output_sptr->operator[](z).get_min_index(); y <= output_sptr->operator[](z).get_max_index(); ++y)
  for (int x = output_sptr->operator[](z)[y].get_min_index(); x <= output_sptr->operator[](z)[y].get_max_index(); ++x)
  {
    val += (*output_sptr)[z][y][x] * (*input_sptr)[z][y][x];
  }
  std::cout << "CUDA gibbs gradient dot input = " << val <<"("<<CUDA_gibbs_prior_sptr->compute_gradient_times_input(*input_sptr,*image_sptr)<<")"<< std::endl;


  val=0;
  original_prior_sptr->compute_gradient(*output_sptr, *image_sptr);
  for (int z = output_sptr->get_min_index(); z <= output_sptr->get_max_index(); ++z)
  for (int y = output_sptr->operator[](z).get_min_index(); y <= output_sptr->operator[](z).get_max_index(); ++y)
  for (int x = output_sptr->operator[](z)[y].get_min_index(); x <= output_sptr->operator[](z)[y].get_max_index(); ++x)
  {
    val += (*output_sptr)[z][y][x] * (*input_sptr)[z][y][x];
  }
  std::cout << "original gradient dot input = " << val << std::endl;

  //HESSIAN TEST
  std::cout<<std::endl;
  double hess,hess_times_inp;
  
  BasicCoordinate<3, int> A = make_coordinate(50, 0, 0);

  gibbs_prior_sptr->compute_Hessian(*output_sptr, A, *image_sptr);
  //gibbs_prior_sptr->accumulate_Hessian_times_input(*output_sptr,*image_sptr,*input_sptr);
  hess=0;
  for (int z = output_sptr->get_min_index(); z <= output_sptr->get_max_index(); ++z)
  for (int y = output_sptr->operator[](z).get_min_index(); y <= output_sptr->operator[](z).get_max_index(); ++y)
  for (int x = output_sptr->operator[](z)[y].get_min_index(); x <= output_sptr->operator[](z)[y].get_max_index(); ++x)
    hess += (*output_sptr)[z][y][x];
  std::cout << "gibbs Hessian sum: " << hess << std::endl;

  CUDA_gibbs_prior_sptr->compute_Hessian(*output_sptr, A, *image_sptr);
  //CUDA_gibbs_prior_sptr->accumulate_Hessian_times_input(*output_sptr,*image_sptr,*input_sptr);
  hess =0;
  for (int z = output_sptr->get_min_index(); z <= output_sptr->get_max_index(); ++z)
  for (int y = output_sptr->operator[](z).get_min_index(); y <= output_sptr->operator[](z).get_max_index(); ++y)
  for (int x = output_sptr->operator[](z)[y].get_min_index(); x <= output_sptr->operator[](z)[y].get_max_index(); ++x)
    hess += (*output_sptr)[z][y][x];
  std::cout << "CUDA gibbs Hessian sum: " << hess << std::endl;

  original_prior_sptr->compute_Hessian(*output_sptr, A, *image_sptr);
  //original_prior_sptr->accumulate_Hessian_times_input(*output_sptr,*image_sptr,*input_sptr);
  hess =0;
  for (int z = output_sptr->get_min_index(); z <= output_sptr->get_max_index(); ++z)
  for (int y = output_sptr->operator[](z).get_min_index(); y <= output_sptr->operator[](z).get_max_index(); ++y)
  for (int x = output_sptr->operator[](z)[y].get_min_index(); x <= output_sptr->operator[](z)[y].get_max_index(); ++x)
    hess += (*output_sptr)[z][y][x];
  std::cout << "original Hessian sum: " << hess << std::endl;
    
  std::cout<<std::endl;
  output_sptr->fill(0.F);
  input_sptr.reset(image_sptr->clone());
  gibbs_prior_sptr->accumulate_Hessian_times_input(*output_sptr,*image_sptr,*input_sptr);
  hess=0;
  for (int z = output_sptr->get_min_index(); z <= output_sptr->get_max_index(); ++z)
  for (int y = output_sptr->operator[](z).get_min_index(); y <= output_sptr->operator[](z).get_max_index(); ++y)
  for (int x = output_sptr->operator[](z)[y].get_min_index(); x <= output_sptr->operator[](z)[y].get_max_index(); ++x)
    hess += (*output_sptr)[z][y][x];
  std::cout << "gibbs Hessian_times_input sum: " << hess << std::endl;

  output_sptr->fill(0.F);
  input_sptr.reset(image_sptr->clone());
  CUDA_gibbs_prior_sptr->accumulate_Hessian_times_input(*output_sptr,*image_sptr,*input_sptr);
  hess=0;
  for (int z = output_sptr->get_min_index(); z <= output_sptr->get_max_index(); ++z)
  for (int y = output_sptr->operator[](z).get_min_index(); y <= output_sptr->operator[](z).get_max_index(); ++y)
  for (int x = output_sptr->operator[](z)[y].get_min_index(); x <= output_sptr->operator[](z)[y].get_max_index(); ++x)
    hess += (*output_sptr)[z][y][x];
  std::cout << "CUDA Hessian_times_input sum: " << hess << std::endl;

  output_sptr->fill(0.F);
  input_sptr.reset(image_sptr->clone());
  original_prior_sptr->accumulate_Hessian_times_input(*output_sptr,*image_sptr,*input_sptr);
  hess=0;
  for (int z = output_sptr->get_min_index(); z <= output_sptr->get_max_index(); ++z)
  for (int y = output_sptr->operator[](z).get_min_index(); y <= output_sptr->operator[](z).get_max_index(); ++y)
  for (int x = output_sptr->operator[](z)[y].get_min_index(); x <= output_sptr->operator[](z)[y].get_max_index(); ++x)
    hess += (*output_sptr)[z][y][x];
  std::cout << "CUDA Hessian_times_input sum: " << hess << std::endl;



  // auto gibbs_rdp = std::make_shared<CudaGibbsRelativeDifferencePrior<float>>(false, 1.F, 2.F, 2);
  // gibbs_rdp->set_up(image_sptr);

  // auto rdp = std::make_shared<CudaRelativeDifferencePrior<float>>(false, 1.F, 2.F, 2);
  // rdp->set_up(image_sptr);

  // std::cout<<"compute_value: "<<gibbs_rdp->compute_value(*image_sptr)<<std::endl;
  // std::cout<<"compute_value: "<<rdp->compute_value(*image_sptr)<<std::endl;

  //   float time_comp_val = Time_it([&]() {

  //   rdp->compute_gradient(*output_sptr, *image_sptr);
  // });
  // std::cout << "timing " << time_comp_val << " ms" << std::endl;




  // time_comp_val = Time_it([&]() {

  //   gibbs_rdp->compute_gradient(*output_sptr, *image_sptr);
  // });
  // std::cout << "gibbs_timings  " << time_comp_val << " ms" << std::endl;



  // auto rdp = std::make_shared<CudaRelativeDifferencePrior<float>>(false, 1.F, 2.F, 2);
  // rdp->set_up(image_sptr);
  // std::cout<<"compute_value: "<<rdp->compute_value(*image_sptr)<<std::endl;

  // time_comp_val = Time_it([&]() {
  //   rdp->compute_value(*image_sptr);
  // });
  // std::cout << "compute_value average time over " << runs << " runs: " << time_comp_val << " ms" << std::endl;


  // gibbs_rdp->compute_Hessian(*output_sptr, A, *image_sptr);
  // auto Gibbs_hessian = *output_sptr;
  // for (int z = output_sptr->get_min_index(); z <= output_sptr->get_max_index(); ++z)
  // for (int y = output_sptr->operator[](z).get_min_index(); y <= output_sptr->operator[](z).get_max_index(); ++y)
  // for (int x = output_sptr->operator[](z)[y].get_min_index(); x <= output_sptr->operator[](z)[y].get_max_index(); ++x)
  // if ((*output_sptr)[z][y][x]  != 0)
  // std::cout<<"[z="<<z<<", y="<<y<<", x="<<x<<"] = "<<(*output_sptr)[z][y][x]<<std::endl;



  // output_sptr.reset(image_sptr->clone());
  // output_sptr->fill(0.F);

  // double gibbs_val = 0.0;
  // auto compute_value_func = [&]() { gibbs_val = prior_sptr->compute_value(*image_sptr); };
  // float avg_ms = Time_it(compute_value_func);
  // std::cout << "compute_value GPU average time over " << runs << " runs: " << avg_ms << " ms" << std::endl;

  // auto quad_prior_sptr = std::make_shared<QuadraticPrior<float>>(false, 1.F);
  // quad_prior_sptr->set_up(image_sptr);
  // BasicCoordinate<3, int> A = make_coordinate(50, 0, 0);
  // quad_prior_sptr->compute_Hessian(*output_sptr, A, *image_sptr);
  // for (int z = output_sptr->get_min_index(); z <= output_sptr->get_max_index(); ++z)
  // for (int y = output_sptr->operator[](z).get_min_index(); y <= output_sptr->operator[](z).get_max_index(); ++y)
  // for (int x = output_sptr->operator[](z)[y].get_min_index(); x <= output_sptr->operator[](z)[y].get_max_index(); ++x)
  // if ((*output_sptr)[z][y][x]  != 0)
  // std::cout<<"[z="<<z<<", y="<<y<<", x="<<x<<"] = "<<(*output_sptr)[z][y][x]<<std::endl;

  // std::cout<<std::endl;
  // auto rdp = std::make_shared<RelativeDifferencePrior<float>>(false, 1.F, 2.F, 2);
  // rdp->set_up(image_sptr);
  // rdp->compute_Hessian(*output_sptr, A, *image_sptr);
  // for (int z = output_sptr->get_min_index(); z <= output_sptr->get_max_index(); ++z)
  // for (int y = output_sptr->operator[](z).get_min_index(); y <= output_sptr->operator[](z).get_max_index(); ++y)
  // for (int x = output_sptr->operator[](z)[y].get_min_index(); x <= output_sptr->operator[](z)[y].get_max_index(); ++x)
  // if ((*output_sptr)[z][y][x]  != 0)
  // std::cout<<"[z="<<z<<", y="<<y<<", x="<<x<<"] = "<<(*output_sptr)[z][y][x] - Gibbs_hessian[z][y][x]<<std::endl;

  return EXIT_SUCCESS;
}
