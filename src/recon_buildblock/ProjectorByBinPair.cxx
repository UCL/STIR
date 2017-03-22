//
//
/*!
  \file
  \ingroup projection

  \brief non-inline implementations for stir::ProjectorByBinPair
  
  \author Kris Thielemans
    
*/
/*
    Copyright (C) 2000- 2009, Hammersmith Imanet Ltd
    This file is part of STIR.

    This file is free software; you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation; either version 2.1 of the License, or
    (at your option) any later version.

    This file is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    See STIR/LICENSE.txt for details
*/


#include "stir/recon_buildblock/ProjectorByBinPair.h"
#include "stir/Succeeded.h"
#include "stir/recon_buildblock/PresmoothingForwardProjectorByBin.h"
#include "stir/recon_buildblock/PostsmoothingBackProjectorByBin.h"
#include "stir/recon_buildblock/ForwardProjectorByBinUsingProjMatrixByBin.h"
#include "stir/recon_buildblock/BackProjectorByBinUsingProjMatrixByBin.h"
#include "stir/is_null_ptr.h"


START_NAMESPACE_STIR


ProjectorByBinPair::
ProjectorByBinPair()
{
}

Succeeded
ProjectorByBinPair::
set_up(const shared_ptr<ProjDataInfo>& proj_data_info_ptr,
       const shared_ptr<DiscretisedDensity<3,float> >& image_info_ptr)

{    	 
  // TODO use return values
  forward_projector_sptr->set_up(proj_data_info_ptr, image_info_ptr);
  back_projector_sptr->set_up(proj_data_info_ptr, image_info_ptr);
  return Succeeded::yes;
}

void
ProjectorByBinPair::
enable_tof(const shared_ptr<ProjDataInfo>& _proj_data_info_sptr, const bool v)
{
    // Check if it is PresmoothingForwardProjectorByBin
    PresmoothingForwardProjectorByBin* forward=
            dynamic_cast<PresmoothingForwardProjectorByBin*> (forward_projector_sptr.get());

    if (!is_null_ptr(forward))
    {
        ForwardProjectorByBinUsingProjMatrixByBin* original_forward =
                dynamic_cast<ForwardProjectorByBinUsingProjMatrixByBin* > (forward->get_original_forward_projector_ptr());

        if (is_null_ptr(original_forward))
            error("Currently only ForwardProjectorByBinUsingProjMatrixByBin supports TOF reconstruction. Abort.");
        else
            original_forward->enable_tof(_proj_data_info_sptr, v);
    }
    else // if is ForwardProjectorByBinUsingProjMatrixByBin directly.
    {
        ForwardProjectorByBinUsingProjMatrixByBin* original_forward =
                dynamic_cast<ForwardProjectorByBinUsingProjMatrixByBin* > (forward_projector_sptr.get());

        if (is_null_ptr(original_forward))
            error("Currently only ForwardProjectorByBinUsingProjMatrixByBin supports TOF reconstruction. Abort.");
        else
            original_forward->enable_tof(_proj_data_info_sptr, v);
    }

//    // Check if it is PostSmoothingBackProjectorByBin is used.
//    PostsmoothingBackProjectorByBin* back=
//            dynamic_cast<PostsmoothingBackProjectorByBin*> (back_projector_sptr.get());

//    if (!is_null_ptr(back))
//    {
//        BackProjectorByBinUsingProjMatrixByBin* original_back =
//                dynamic_cast<BackProjectorByBinUsingProjMatrixByBin* > (back->get_original_back_projector_ptr());

//        if (is_null_ptr(original_back))
//            error("Currently only ForwardProjectorByBinUsingProjMatrixByBin supports TOF reconstruction. Abort.");
//        else
//            original_back->enable_tof(_proj_data_info_sptr, v);
//    }
//    else // if is ForwardProjectorByBinUsingProjMatrixByBin directly.
//    {
//        BackProjectorByBinUsingProjMatrixByBin* original_back =
//                dynamic_cast<BackProjectorByBinUsingProjMatrixByBin* > (back_projector_sptr.get());

//        if (is_null_ptr(original_back))
//            error("Currently only BackProjectorByBinUsingProjMatrixByBin supports TOF reconstruction. Abort.");
//        else
//            original_back->enable_tof(_proj_data_info_sptr, v);
//    }

}

void
ProjectorByBinPair::
set_tof_data(const CartesianCoordinate3D<float>* _point1,
             const CartesianCoordinate3D<float>* _point2)
{
    // Check if it is PresmoothingForwardProjectorByBin
    PresmoothingForwardProjectorByBin* forward=
            dynamic_cast<PresmoothingForwardProjectorByBin*> (forward_projector_sptr.get());

    if (!is_null_ptr(forward))
    {
        ForwardProjectorByBinUsingProjMatrixByBin* original_forward =
                dynamic_cast<ForwardProjectorByBinUsingProjMatrixByBin* > (forward->get_original_forward_projector_ptr());

        if (is_null_ptr(original_forward))
            error("Currently only ForwardProjectorByBinUsingProjMatrixByBin supports TOF reconstruction. Abort.");
        else
        {
            original_forward->set_tof_data(_point1, _point2);

            PostsmoothingBackProjectorByBin* back=
                    dynamic_cast<PostsmoothingBackProjectorByBin*> (back_projector_sptr.get());

            if (!is_null_ptr(back))
            {
                BackProjectorByBinUsingProjMatrixByBin* original_back =
                        dynamic_cast<BackProjectorByBinUsingProjMatrixByBin* > (back->get_original_back_projector_ptr());

                if (is_null_ptr(original_back))
                    error("Currently only ForwardProjectorByBinUsingProjMatrixByBin supports TOF reconstruction. Abort.");
                else
                    original_back->enable_tof(original_forward->get_tof_row());
            }
            else // if is ForwardProjectorByBinUsingProjMatrixByBin directly.
            {
                BackProjectorByBinUsingProjMatrixByBin* original_back =
                        dynamic_cast<BackProjectorByBinUsingProjMatrixByBin* > (back_projector_sptr.get());

                if (is_null_ptr(original_back))
                    error("Currently only BackProjectorByBinUsingProjMatrixByBin supports TOF reconstruction. Abort.");
                else
                    original_back->enable_tof(original_forward->get_tof_row());
            }
        }
    }
    else // if is ForwardProjectorByBinUsingProjMatrixByBin directly.
    {
        ForwardProjectorByBinUsingProjMatrixByBin* original_forward =
                dynamic_cast<ForwardProjectorByBinUsingProjMatrixByBin* > (forward_projector_sptr.get());

        if (is_null_ptr(original_forward))
            error("Currently only ForwardProjectorByBinUsingProjMatrixByBin supports TOF reconstruction. Abort.");
        else
        {
            original_forward->set_tof_data(_point1, _point2);

            PostsmoothingBackProjectorByBin* back=
                    dynamic_cast<PostsmoothingBackProjectorByBin*> (back_projector_sptr.get());

            if (!is_null_ptr(back))
            {
                BackProjectorByBinUsingProjMatrixByBin* original_back =
                        dynamic_cast<BackProjectorByBinUsingProjMatrixByBin* > (back->get_original_back_projector_ptr());

                if (is_null_ptr(original_back))
                    error("Currently only ForwardProjectorByBinUsingProjMatrixByBin supports TOF reconstruction. Abort.");
                else
                    original_back->enable_tof(original_forward->get_tof_row());
            }
            else // if is ForwardProjectorByBinUsingProjMatrixByBin directly.
            {
                BackProjectorByBinUsingProjMatrixByBin* original_back =
                        dynamic_cast<BackProjectorByBinUsingProjMatrixByBin* > (back_projector_sptr.get());

                if (is_null_ptr(original_back))
                    error("Currently only BackProjectorByBinUsingProjMatrixByBin supports TOF reconstruction. Abort.");
                else
                    original_back->enable_tof(original_forward->get_tof_row());
            }
        }
    }
}

ProjMatrixElemsForOneBin*
ProjectorByBinPair::
get_current_tof_row() const
{
    // Check if it is PresmoothingForwardProjectorByBin
    PresmoothingForwardProjectorByBin* forward=
            dynamic_cast<PresmoothingForwardProjectorByBin*> (forward_projector_sptr.get());

    if (!is_null_ptr(forward))
    {
        ForwardProjectorByBinUsingProjMatrixByBin* original_forward =
                dynamic_cast<ForwardProjectorByBinUsingProjMatrixByBin* > (forward->get_original_forward_projector_ptr());

        if (is_null_ptr(original_forward))
            error("Currently only ForwardProjectorByBinUsingProjMatrixByBin supports TOF reconstruction. Abort.");
        else
            return original_forward->get_tof_row();
    }
    else // if is ForwardProjectorByBinUsingProjMatrixByBin directly.
    {
        ForwardProjectorByBinUsingProjMatrixByBin* original_forward =
                dynamic_cast<ForwardProjectorByBinUsingProjMatrixByBin* > (forward_projector_sptr.get());

        if (is_null_ptr(original_forward))
            error("Currently only ForwardProjectorByBinUsingProjMatrixByBin supports TOF reconstruction. Abort.");
        else
           return original_forward->get_tof_row();
    }
}

//ForwardProjectorByBin const * 
const shared_ptr<ForwardProjectorByBin>
ProjectorByBinPair::
get_forward_projector_sptr() const
{
  return forward_projector_sptr;
}

//BackProjectorByBin const * 
const shared_ptr<BackProjectorByBin>
ProjectorByBinPair::
get_back_projector_sptr() const
{
  return back_projector_sptr;
}

END_NAMESPACE_STIR
