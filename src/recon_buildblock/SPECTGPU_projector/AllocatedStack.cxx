//
//
/*!

  \file
  \ingroup projection
  \ingroup SPECTGPU

  \brief methods for allocating or deallocating arrays in the allocated stack

  \author Daniel Deidda


*/
/*
    Copyright (C) 2026, National Physical Laboratory
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details
*/
#include "stir/recon_buildblock/SPECTGPU_projector/AllocatedStack.h"
#include "stir/cuda_utilities.h"
START_NAMESPACE_STIR
namespace SPECTGPU
{
void
AllocatedStack::allocate(size_t image_size_, size_t sino_size_)
{
  if (image_size == image_size_ && sino_size == sino_size_)
    return;

  image_size = image_size_;
  sino_size = sino_size_;

  cu_allocate(out_im, image_size);
  cu_allocate(out_umap, image_size);
  cu_allocate(blurred_im, image_size);
  cu_allocate(rotated_im, image_size);
  cu_allocate(rotated_umap, image_size);
  cu_allocate(dev_image, image_size);
  cu_allocate(dev_umap, image_size);
  cu_allocate(dev_sino, sino_size);
}

void
AllocatedStack::deallocate()
{
  cu_free(out_im);
  cu_free(out_umap);
  cu_free(blurred_im);
  cu_free(rotated_im);
  cu_free(rotated_umap);
  cu_free(dev_image);
  cu_free(dev_umap);
  cu_free(dev_sino);

  image_size = 0;
  sino_size = 0;
}

bool
AllocatedStack::is_allocated() const
{
  return image_size != 0;
}

AllocatedStack::~AllocatedStack()
{
  deallocate();
}
} // namespace SPECTGPU
END_NAMESPACE_STIR
