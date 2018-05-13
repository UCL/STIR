/*
 *  Copyright (C) 2015, 2016 University of Leeds
    Copyright (C) 2016, UCL
    Copyright (C) 2018, University of Hull
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

START_NAMESPACE_STIR

inline void CListEventROOT::init_from_data(const int& _ring1, const int& _ring2,
                                    const int& crystal1, const int& crystal2)
{
//    if  (crystal1 < 0 )
//        det1 = scanner_sptr->get_num_detectors_per_ring() + crystal1;
//    else if ( crystal1 >= scanner_sptr->get_num_detectors_per_ring())
//        det1 = crystal1 - scanner_sptr->get_num_detectors_per_ring();
//    else
//        det1 = crystal1;

//    if  (crystal2 < 0 )
//        det2 = scanner_sptr->get_num_detectors_per_ring() + crystal2;
//    else if ( crystal2 >= scanner_sptr->get_num_detectors_per_ring())
//        det2 = crystal2 - scanner_sptr->get_num_detectors_per_ring();
//    else
//        det2 = crystal2;

    // STIR assumes that 0 is on y whill GATE on the x axis
    det1 = crystal1 + quarter_of_detectors;
    det2 = crystal2 + quarter_of_detectors;

    if  (det1 < 0 )
        det1 = scanner_sptr->get_num_detectors_per_ring() + det1;
    else if ( det1 >= scanner_sptr->get_num_detectors_per_ring())
        det1 = det1 - scanner_sptr->get_num_detectors_per_ring();

    if  (det2 < 0 )
        det2 = scanner_sptr->get_num_detectors_per_ring() + det2;
    else if ( det2 >= scanner_sptr->get_num_detectors_per_ring())
        det2 = det2 - scanner_sptr->get_num_detectors_per_ring();

    if (det1 > det2)
    {
        int tmp = det1;
        det1 = det2;
        det2 = tmp;

        ring1 = _ring2;
        ring2 = _ring1;
        swapped = true;
    }
    else
    {
        ring1 = _ring1;
        ring2 = _ring2;
        swapped = false;
    }
}

END_NAMESPACE_STIR

