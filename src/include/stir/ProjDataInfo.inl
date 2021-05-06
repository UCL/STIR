//
//
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000 - 2011-10-14, Hammersmith Imanet Ltd
    Copyright (C) 2011-07-01 - 2011, Kris Thielemans
    Copyright (C) 2016, 2020, University College London
    This file is part of STIR.

    SPDX-License-Identifier: Apache-2.0 AND License-ref-PARAPET-license
    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup projdata
  \brief Implementations of inline functions for class stir::ProjDataInfo

  \author Sanida Mustafovic
  \author Kris Thielemans
  \author Nikos Efthimiou
  \author PARAPET project

*/

START_NAMESPACE_STIR

shared_ptr<ProjDataInfo> 
ProjDataInfo::
create_shared_clone() const
{
  shared_ptr<ProjDataInfo> sptr(this->clone());
  return sptr;
}

int 
ProjDataInfo::get_num_segments() const
{ return (max_axial_pos_per_seg.get_length());}


int
ProjDataInfo::get_num_axial_poss(const int segment_num) const
{ return  max_axial_pos_per_seg[segment_num] - min_axial_pos_per_seg[segment_num]+1;}

int 
ProjDataInfo::get_num_views() const
{ return max_view_num - min_view_num + 1; }

int 
ProjDataInfo::get_num_tangential_poss() const
{ return  max_tangential_pos_num - min_tangential_pos_num + 1; }

int
ProjDataInfo::get_num_tof_poss() const
{ return 1; /* always 1 at the moment */ }

int 
ProjDataInfo::get_min_segment_num() const
{ return (max_axial_pos_per_seg.get_min_index()); }

int 
ProjDataInfo::get_max_segment_num()const
{ return (max_axial_pos_per_seg.get_max_index());  }

int
ProjDataInfo::get_min_axial_pos_num(const int segment_num) const
{ return min_axial_pos_per_seg[segment_num];}


int
ProjDataInfo::get_max_axial_pos_num(const int segment_num) const
{ return max_axial_pos_per_seg[segment_num];}


int 
ProjDataInfo::get_min_view_num() const
  { return min_view_num; }

int 
ProjDataInfo::get_max_view_num()  const
{ return max_view_num; }


int 
ProjDataInfo::get_min_tangential_pos_num()const
{ return min_tangential_pos_num; }

int 
ProjDataInfo::get_max_tangential_pos_num()const
{ return max_tangential_pos_num; }

float 
ProjDataInfo::get_costheta(const Bin& bin) const
{
  return
    1/sqrt(1+square(get_tantheta(bin)));
}

float
ProjDataInfo::get_m(const Bin& bin) const
{
  return 
    get_t(bin)/get_costheta(bin);
}

const 
Scanner*
ProjDataInfo::get_scanner_ptr() const
{ 
  return scanner_ptr.get();
}

shared_ptr<Scanner>
ProjDataInfo::get_scanner_sptr() const
{
  return scanner_ptr;
}


int
ProjDataInfo::get_num_non_tof_sinograms() const
{
  int num_sinos = 0;
  for (int s=this->get_min_segment_num(); s<= this->get_max_segment_num(); ++s)
    num_sinos += this->get_num_axial_poss(s);

  return num_sinos;
}

int
ProjDataInfo::get_num_sinograms() const
{
    return this->get_num_non_tof_sinograms()*this->get_num_tof_poss();
}

std::size_t
ProjDataInfo::size_all() const
{ return static_cast<std::size_t>(this->get_num_sinograms()) *
    static_cast<std::size_t>(this->get_num_views() * this->get_num_tangential_poss()); }

END_NAMESPACE_STIR

