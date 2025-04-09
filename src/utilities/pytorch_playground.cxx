//
//

/*!
\file
\ingroup utilities
\brief this executable is not meant to do something specific, other than facilitate the developlement of the Pytorch interface.
Heavily inspired by compare_images.cxxs

\author Nikos Efthimiou
*/


#include "stir/DiscretisedDensity.h"
#include "stir/ArrayFunction.h"
#include "stir/recon_array_functions.h"
#include "stir/IO/read_from_file.h"
#include "stir/is_null_ptr.h"
#include "stir/warning.h"
#include <numeric>
#include <stdlib.h>
#include <chrono> // For timing
#include <cuda_runtime.h>
#include <stir/TensorWrapper.h>

using std::cerr;
using std::cout;
using std::endl;

USING_NAMESPACE_STIR

//********************** main

int
main(int argc, char* argv[])
{
  if (argc < 3 || argc > 7)
    {
      cerr << "Usage: \n"
           << argv[0] << "\n\t"
           << "[-r rimsize] \n\t"
           << "[-t tolerance] \n\t"
           << "old_image new_image \n\t"
           << "'rimsize' has to be a nonnegative integer.\n\t"
           << "'tolerance' is by default .0005 \n\t"
           << "When the -r option is used, the (radial) rim of the\n\t"
           << "images will be set to 0, for 'rimsize' pixels.\n";
      return (EXIT_FAILURE);
    }
  // skip program name
  --argc;
  ++argv;
  int rim_truncation_image = -1;
  float tolerance = .0005F;

  int device_count = 0;
  cudaError_t err = cudaGetDeviceCount(&device_count);

  if (err != cudaSuccess) {
      std::cerr << "CUDA error: " << cudaGetErrorString(err) << std::endl;
      return -1;
    }

  if (device_count == 0) {
      std::cout << "No CUDA devices available." << std::endl;
      return 0;
    }

  std::cout << "Number of CUDA devices: " << device_count << std::endl;

  for (int device = 0; device < device_count; ++device) {
      cudaDeviceProp device_prop;
      cudaGetDeviceProperties(&device_prop, device);

      std::cout << "Device " << device << ": " << device_prop.name << std::endl;
      std::cout << "  CUDA Capability: " << device_prop.major << "." << device_prop.minor << std::endl;
      std::cout << "  Total Memory: " << device_prop.totalGlobalMem / (1024 * 1024) << " MB" << std::endl;
      std::cout << "  Multiprocessors: " << device_prop.multiProcessorCount << std::endl;
      std::cout << "  Clock Rate: " << device_prop.clockRate / 1000 << " MHz" << std::endl;
    }

  // first process command line options
  while (argc > 0 && argv[0][0] == '-')
    {
      if (strcmp(argv[0], "-r") == 0)
        {
          if (argc < 2)
            {
              cerr << "Option '-r' expects a nonnegative (integer) argument\n";
              exit(EXIT_FAILURE);
            }
          rim_truncation_image = atoi(argv[1]);
          argc -= 2;
          argv += 2;
        }
      if (strcmp(argv[0], "-t") == 0)
        {
          if (argc < 2)
            {
              cerr << "Option '-t' expects a (float) argument\n";
              exit(EXIT_FAILURE);
            }
          tolerance = static_cast<float>(atof(argv[1]));
          argc -= 2;
          argv += 2;
        }
    }

  shared_ptr<DiscretisedDensity<3, float>> first_operand(read_from_file<DiscretisedDensity<3, float>>(argv[0]));

  if (is_null_ptr(first_operand))
    {
      cerr << "Could not read first file\n";
      exit(EXIT_FAILURE);
    }

  stir::IndexRange d = first_operand->get_index_range();

  std::cout << "D:" << d.get_min_index() << " " << d.get_max_index() << " " << d.get_length() << std::endl;
  std::cout << "D:" << d[0].get_min_index() << " " << d[0].get_max_index() << " " << d[0].get_length() << std::endl;
  std::cout << "G:" << d[0][0].get_min_index() << " " << d[0][0].get_max_index() << " " << d[0][0].get_length() << std::endl;

  std::vector<int64_t> shape({d.get_length(), d[0].get_length(), d[0][0].get_length()});
  // Allocate Tensor
  // TensorWrapper<3, float> tw(shape);
  TensorWrapper tw((*first_operand));

  // Print the tensor's device
  tw.print_device();
  // Move the tensor to the GPU (if available)
  try {
      tw.to_gpu();
      std::cout << "Moved tw to GPU." << std::endl;
    } catch (const std::runtime_error& e) {
      std::cerr << e.what() << std::endl;
    }
  // Print the tensor's device again
  tw.print_device();

  std::cout <<  " H " << std::endl;
  tw.printSizes();
  std::cout <<  " H2 " << std::endl;

  const float a = 1000;
  const float b = 5000;
  // Start timing
  {
    auto start = std::chrono::high_resolution_clock::now();
    std::cout << "stir::Array MAX value: " << (*first_operand).find_max() << std::endl;
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "stir::Array: Time to compute max value: " << duration << " ms" << std::endl;
    std::cout << "\n" << std::endl;
  }
  {
    auto start = std::chrono::high_resolution_clock::now();
    std::cout << "Tensored: MAX value: " << tw.find_max() << std::endl;
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Tensored: Time to compute max value: " << duration << " ms" << std::endl;
    std::cout << "\n" << std::endl;
  }

  {
    auto start = std::chrono::high_resolution_clock::now();
    std::cout << "stir::Array SUM value: " << (*first_operand).sum() << std::endl;
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "stir::Array: Time to compute SUM value: " << duration << " ms" << std::endl;
    std::cout << "\n" << std::endl;
  }
  {
    auto start = std::chrono::high_resolution_clock::now();
    std::cout << "Tensored: SUM value: " << tw.sum() << std::endl;
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Tensored: Time to compute SUM value: " << duration << " ms" << std::endl;
    std::cout << "\n" << std::endl;
  }

  {
    auto start = std::chrono::high_resolution_clock::now();
    std::cout << "stir::Array SUM_pos value: " << (*first_operand).sum_positive() << std::endl;
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "stir::Array: Time to compute SUM_pos value: " << duration << " ms" << std::endl;
    std::cout << "\n" << std::endl;
  }
  {
    auto start = std::chrono::high_resolution_clock::now();
    std::cout << "Tensored: SUM_pos value: " << tw.sum_positive() << std::endl;
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Tensored: Time to compute SUM_pos value: " << duration << " ms" << std::endl;
    std::cout << "\n" << std::endl;
  }

  std::cout << "____________XAPYB__________" << std::endl;
  {
    auto cloned_empty_first_operand = first_operand->get_empty_copy();
    auto cloned_first_operand = first_operand->clone();
    auto start = std::chrono::high_resolution_clock::now();
    cloned_empty_first_operand->xapyb((*first_operand), a, *cloned_first_operand, b);
    std::cout << "stir::Array: MAX value after xapyb: " << cloned_empty_first_operand->find_max() << std::endl;
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "stir::Array: Time to compute max value after xapyb: : " << duration << " ms" << std::endl;
    std::cout << "\n" << std::endl;
  }

  {
    auto cloned_empty_tw = tw.get_empty_copy();
    cloned_empty_tw->print_device();
    try {
        cloned_empty_tw->to_gpu();
        std::cout << "Moved cloned_empty_tw to GPU." << std::endl;
      } catch (const std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
      }
    // Print the tensor's device again
    cloned_empty_tw->print_device();

    auto cloned_tw = tw.clone();
    cloned_tw->print_device();
    try {
        cloned_tw->to_gpu();
        std::cout << "Moved cloned_tw to GPU." << std::endl;
      } catch (const std::runtime_error& e) {
        std::cerr << e.what() << std::endl;
      }
    // Print the tensor's device again
    cloned_tw->print_device();

    auto start = std::chrono::high_resolution_clock::now();
    cloned_empty_tw->xapyb(tw, a, *cloned_tw, b);
    std::cout << "Tensored: MAX value after xapyb: " << cloned_empty_tw->find_max() << std::endl;
    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "Tensored: Time to compute max value: " << duration << " ms" << std::endl;
    std::cout << "\n" << std::endl;
  }

         // {
         //   auto start = std::chrono::high_resolution_clock::now();
         //   std::cout << "Tensored first_operand MAX value: " << tw.find_max() << std::endl;
         //   auto end = std::chrono::high_resolution_clock::now();
         //   auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
         //   std::cout << "Time to compute max value: " << duration << " ms" << std::endl;
         // }


  std::cout << "STOP HERE" << std::endl;
  return 1;

  // shared_ptr<DiscretisedDensity<3, float>> second_operand(read_from_file<DiscretisedDensity<3, float>>(argv[1]));
  // if (is_null_ptr(second_operand))
  //   {
  //     cerr << "Could not read 2nd file\n";
  //     exit(EXIT_FAILURE);
  //   }

  //        // check if images are compatible
  // {
  //   std::string explanation;
  //   if (!first_operand->has_same_characteristics(*second_operand, explanation))
  //     {
  //       warning("input images do not have the same characteristics.\n%s", explanation.c_str());
  //       return EXIT_FAILURE;
  //     }
  // }

  // if (rim_truncation_image >= 0)
  //   {
  //     truncate_rim(*first_operand, rim_truncation_image);
  //     truncate_rim(*second_operand, rim_truncation_image);
  //   }

  // float reference_max = first_operand->find_max();
  // float reference_min = first_operand->find_min();

  // float amplitude = fabs(reference_max) > fabs(reference_min) ? fabs(reference_max) : fabs(reference_min);

  // *first_operand -= *second_operand;
  // const float max_error = first_operand->find_max();
  // const float min_error = first_operand->find_min();
  // in_place_abs(*first_operand);
  // const float max_abs_error = first_operand->find_max();

  // const bool same = (max_abs_error / amplitude <= tolerance);

  // cout << "\nMaximum absolute error = " << max_abs_error << "\nMaximum in (1st - 2nd) = " << max_error
  //      << "\nMinimum in (1st - 2nd) = " << min_error << endl;
  // cout << "Error relative to sup-norm of first image = " << (max_abs_error / amplitude) * 100 << " %" << endl;

  // cout << "\nImage arrays ";

  // if (same)
  //   {
  //     cout << (max_abs_error == 0 ? "are " : "deemed ") << "identical\n";
  //   }
  // else
  //   {
  //     cout << "deemed different\n";
  //   }
  // cout << "(tolerance used: " << tolerance * 100 << " %)\n\n";
  // return same ? EXIT_SUCCESS : EXIT_FAILURE;

} // end main
