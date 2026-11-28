#ifndef __stir_gpu_ALLOCATEDSTACK_H
#define __stir_gpu_ALLOCATEDSTACK_H
/*!
\file
\ingroup projection
\ingroup SPECTGPU

\brief  SPECTGPU's allocating structure to make sure we do not make allocation in big loops.

\author Daniel Deidda
*/

#include "stir/common.h"

START_NAMESPACE_STIR
namespace SPECTGPU
{
struct AllocatedStack
{

  size_t image_size = 0;
  size_t sino_size = 0;

  float* dev_image = nullptr;
  float* dev_umap = nullptr;
  float* out_im = nullptr;
  float* out_umap = nullptr;
  float* dev_sino = nullptr;
  float* blurred_im = nullptr;
  float* rotated_im = nullptr;
  float* rotated_umap = nullptr;

  void allocate(size_t image_size_, size_t sino_size_);

  void deallocate();

  bool is_allocated() const;

  ~AllocatedStack();
};
} // namespace SPECTGPU
END_NAMESPACE_STIR
#endif // ALLOCATEDSTACK_H
