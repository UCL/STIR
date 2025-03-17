/*
    Copyright (C) 2022, Matthew Strugari
    Copyright (C) 2014, Biomedical Image Group (GIB), Universitat de Barcelona, Barcelona, Spain. All rights reserved.
    Copyright (C) 2014, 2021, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0

    See STIR/LICENSE.txt for details

    \author Carles Falcon
    \author Matthew Strugari
*/

#ifndef _WEIGHT3D_SPECT_mph_H
#define _WEIGHT3D_SPECT_mph_H

namespace SPECTUB_mph
{

void wm_calculation_mph(bool do_estim,
                        const int kOS,
                        psf2d_type* psf2d_bin,
                        psf2d_type* psf_subs,
                        psf2d_type* psf2d_aux,
                        const psf2d_type* kern,
                        const float* attmap,
                        const bool* msk_3d,
                        int* Nitems,
                        const wmh_mph_type& wmh,
                        wm_da_type& wm,
                        const pcf_type& pcf);

void fill_psfi(psf2d_type* kern, const wmh_mph_type& wmh);

} // namespace SPECTUB_mph

#endif
