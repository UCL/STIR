//
//
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000 - 2009-05-13, Hammersmith Imanet Ltd
    Copyright (C) 2011-07-01 - 2011, Kris Thielemans
    Copyright (C) 2018, University College London
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
/*!
  \file
  \ingroup projdata

  \brief Implementation of non-inline functions of class stir::ProjDataInfo

  \author Sanida Mustafovic
  \author Kris Thielemans
  \author PARAPET project

*/
#include "stir/ProjDataInfo.h"
#include "stir/ProjDataInfoCylindricalArcCorr.h"
#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "stir/Scanner.h"
#include "stir/Viewgram.h"
#include "stir/Sinogram.h"
#include "stir/SegmentBySinogram.h"
#include "stir/SegmentByView.h"
#include "stir/RelatedViewgrams.h"
#include "stir/ViewSegmentNumbers.h"
#include "stir/Coordinate2D.h"
#include "stir/Coordinate3D.h"
#include "stir/IndexRange2D.h"
#include "stir/IndexRange3D.h"
#include "stir/Bin.h"
// include for ask and ask_num
#include "stir/utilities.h"

#include <iostream>
#include <typeinfo>
#include <vector>
#include <algorithm>
#ifdef BOOST_NO_STRINGSTREAM
#include <strstream.h>
#else
#include <sstream>
#endif
#include "boost/format.hpp"

#ifndef STIR_NO_NAMESPACES
using std::string;
using std::vector;
using std::cerr;
using std::cout;
using std::endl;
using std::equal;
#endif

START_NAMESPACE_STIR


float 
ProjDataInfo::get_sampling_in_t(const Bin& bin) const
{
  return
    (get_t(Bin(bin.segment_num(), bin.view_num(), bin.axial_pos_num()+1,bin.tangential_pos_num())) -
     get_t(Bin(bin.segment_num(), bin.view_num(), bin.axial_pos_num()-1,bin.tangential_pos_num()))
     )/2;
}

float 
ProjDataInfo::get_sampling_in_m(const Bin& bin) const
{
  return
    (get_m(Bin(bin.segment_num(), bin.view_num(), bin.axial_pos_num()+1,bin.tangential_pos_num())) -
     get_m(Bin(bin.segment_num(), bin.view_num(), bin.axial_pos_num()-1,bin.tangential_pos_num()))
     )/2;
}


float 
ProjDataInfo::get_sampling_in_s(const Bin& bin) const
{
  return
    (get_s(Bin(bin.segment_num(), bin.view_num(), bin.axial_pos_num(),bin.tangential_pos_num()+1)) -
     get_s(Bin(bin.segment_num(), bin.view_num(), bin.axial_pos_num(),bin.tangential_pos_num()-1))
     )/2;
}

void 
ProjDataInfo::set_num_views(const int num_views)
{
  min_view_num = 0;
  max_view_num = num_views-1;
}

void 
ProjDataInfo::set_num_tangential_poss(const int num_tang_poss)
{

  min_tangential_pos_num = -(num_tang_poss/2);
  max_tangential_pos_num = min_tangential_pos_num + num_tang_poss-1;
}

/*! Sets min_axial_pos_per_seg to 0 */
void 
ProjDataInfo::set_num_axial_poss_per_segment(const VectorWithOffset<int>& num_axial_poss_per_segment)
{
  // first do assignments to make the members the correct size 
  // (data will be overwritten)
  min_axial_pos_per_seg= num_axial_poss_per_segment;
  max_axial_pos_per_seg= num_axial_poss_per_segment;
  
  for (int i=num_axial_poss_per_segment.get_min_index(); 
       i<=num_axial_poss_per_segment.get_max_index();
       i++)
  {
    min_axial_pos_per_seg[i]=0;
    max_axial_pos_per_seg[i]=num_axial_poss_per_segment[i]-1;
  }
    
}

/*! No checks are done on validity of the min_ax_pos_num argument */
void 
ProjDataInfo::set_min_axial_pos_num(const int min_ax_pos_num, const int segment_num)
{
  min_axial_pos_per_seg[segment_num] = min_ax_pos_num;
}
/*! No checks are done on validity of the max_ax_pos_num argument */
void 
ProjDataInfo::set_max_axial_pos_num(const int max_ax_pos_num, const int segment_num)
{
  max_axial_pos_per_seg[segment_num] = max_ax_pos_num;
}


void
ProjDataInfo::set_min_tangential_pos_num(const int min_tang_poss)
{
  min_tangential_pos_num = min_tang_poss;
}

void
ProjDataInfo::set_max_tangential_pos_num(const int max_tang_poss)
{
  max_tangential_pos_num = max_tang_poss;
}


ProjDataInfo::ProjDataInfo()
{}




ProjDataInfo::ProjDataInfo(const shared_ptr<Scanner>& scanner_ptr_v,
                           const VectorWithOffset<int>& num_axial_pos_per_segment_v, 
                           const int num_views_v, 
                           const int num_tangential_poss_v)
			   :scanner_ptr(scanner_ptr_v)

{ 
  set_num_views(num_views_v);
  set_num_tangential_poss(num_tangential_poss_v);
  set_num_axial_poss_per_segment(num_axial_pos_per_segment_v);
}

string
ProjDataInfo::parameter_info()  const
{

#ifdef BOOST_NO_STRINGSTREAM
  // dangerous for out-of-range, but 'old-style' ostrstream seems to need this
  char str[30000];
  ostrstream s(str, 30000);
#else
  std::ostringstream s;
#endif
  s << scanner_ptr->parameter_info();

  s << "\nSegment_num range:           ("
      << get_min_segment_num()
      << ", " <<  get_max_segment_num() << ")\n";
  s << "Number of Views:                " << get_num_views() << endl;
  s << "Number of axial positions per seg: {";
  for (int seg_num=get_min_segment_num(); seg_num <= get_max_segment_num(); ++seg_num)
    s << get_num_axial_poss(seg_num) << " ";
  s << "}\n";
  s << "Number of tangential positions: " << get_num_tangential_poss() << endl;

  return s.str();

}


void
ProjDataInfo::
reduce_segment_range(const int min_segment_num, const int max_segment_num)
{
  assert(min_segment_num >= get_min_segment_num());
  assert(max_segment_num <= get_max_segment_num());

  VectorWithOffset<int> new_min_axial_pos_per_seg(min_segment_num, max_segment_num); 
  VectorWithOffset<int> new_max_axial_pos_per_seg(min_segment_num, max_segment_num);

  for (int segment_num = min_segment_num; segment_num<= max_segment_num; ++segment_num)
  {
    new_min_axial_pos_per_seg[segment_num] =
      min_axial_pos_per_seg[segment_num];
    new_max_axial_pos_per_seg[segment_num] =
      max_axial_pos_per_seg[segment_num];
  }

  min_axial_pos_per_seg = new_min_axial_pos_per_seg;
  max_axial_pos_per_seg = new_max_axial_pos_per_seg;
  
}


Viewgram<float> 
ProjDataInfo::get_empty_viewgram(const int view_num, 
				 const int segment_num, 
				 const bool make_num_tangential_poss_odd) const
{  
  // we can't access the shared ptr, so we have to clone 'this'.
  shared_ptr<ProjDataInfo>  proj_data_info_sptr(this->clone());
  
  if (make_num_tangential_poss_odd && (get_num_tangential_poss()%2==0))
    proj_data_info_sptr->set_max_tangential_pos_num(get_max_tangential_pos_num() + 1);
  
  Viewgram<float> v(proj_data_info_sptr, view_num, segment_num);
   
  return v;
}


Sinogram<float>
ProjDataInfo::get_empty_sinogram(const int axial_pos_num, const int segment_num,
                                      const bool make_num_tangential_poss_odd) const
{
  // we can't access the shared ptr, so we have to clone 'this'.
  shared_ptr<ProjDataInfo>  proj_data_info_sptr(this->clone());
  
  if (make_num_tangential_poss_odd && (get_num_tangential_poss()%2==0))
    proj_data_info_sptr->set_max_tangential_pos_num(get_max_tangential_pos_num() + 1);

  Sinogram<float> s(proj_data_info_sptr, axial_pos_num, segment_num);
   
  return s;
}

SegmentBySinogram<float>
ProjDataInfo::get_empty_segment_by_sinogram(const int segment_num, 
      const bool make_num_tangential_poss_odd) const
{
  assert(segment_num >= get_min_segment_num());
  assert(segment_num <= get_max_segment_num());
 
  // we can't access the shared ptr, so we have to clone 'this'.
  shared_ptr<ProjDataInfo>  proj_data_info_sptr(this->clone());
  
  if (make_num_tangential_poss_odd && (get_num_tangential_poss()%2==0))
    proj_data_info_sptr->set_max_tangential_pos_num(get_max_tangential_pos_num() + 1);

  SegmentBySinogram<float> s(proj_data_info_sptr, segment_num);

  return s;
}  


SegmentByView<float>
ProjDataInfo::get_empty_segment_by_view(const int segment_num, 
				   const bool make_num_tangential_poss_odd) const
{
  assert(segment_num >= get_min_segment_num());
  assert(segment_num <= get_max_segment_num());

    // we can't access the shared ptr, so we have to clone 'this'.
  shared_ptr<ProjDataInfo>  proj_data_info_sptr(this->clone());
  
  if (make_num_tangential_poss_odd && (get_num_tangential_poss()%2==0))
    proj_data_info_sptr->set_max_tangential_pos_num(get_max_tangential_pos_num() + 1);
  
  SegmentByView<float> s(proj_data_info_sptr, segment_num);

  return s;
}

RelatedViewgrams<float> 
ProjDataInfo::get_empty_related_viewgrams(const ViewSegmentNumbers& view_segmnet_num,
                   //const int view_num, const int segment_num,
		   const shared_ptr<DataSymmetriesForViewSegmentNumbers>& symmetries_used,
		   const bool make_num_tangential_poss_odd) const
{
  vector<ViewSegmentNumbers> pairs;
  symmetries_used->get_related_view_segment_numbers(
                                                    pairs, 
                                                    ViewSegmentNumbers(view_segmnet_num.view_num(),view_segmnet_num.segment_num())
    );

  vector<Viewgram<float> > viewgrams;
  viewgrams.reserve(pairs.size());

  for (unsigned int i=0; i<pairs.size(); i++)
  {
    // TODO optimise to get shared proj_data_info_ptr
    viewgrams.push_back(get_empty_viewgram(pairs[i].view_num(),
                                          pairs[i].segment_num(), make_num_tangential_poss_odd));
  }

  return RelatedViewgrams<float>(viewgrams, symmetries_used);
}


/****************** static members **********************/

ProjDataInfo*
ProjDataInfo::ProjDataInfoCTI(const shared_ptr<Scanner>& scanner,
			      const int span, const int max_delta, 
			      const int num_views, const int num_tangential_poss,
                              const bool arc_corrected)
{
  const int num_ring = scanner->get_num_rings();
  if (max_delta > num_ring - 1)
    error(boost::format("construct_proj_data_info: max_ring_difference %d is too large, number of rings is %d")
          % max_delta % num_ring);
	if (span < 1)
    error(boost::format("construct_proj_data_info: span %d has to be larger than 0") % span);
  if (span > 2 * num_ring - 1)
    error(boost::format("construct_proj_data_info: span %d is too large for a scanner with %d rings")
      % span % num_ring);
  if (max_delta<(span-1)/2)
    error(boost::format("construct_proj_data_info: max_ring_difference %d has to be at least (span-1)/2, span is %d")
	        % max_delta % span);

 // Construct first a temporary list of min and max ring diff per segment (0,1,2,3...)
  vector <int> RDmintmp(num_ring);
  vector <int> RDmaxtmp(num_ring);
  
  if (span%2 == 1)
    {
      RDmintmp[0] = -((span-1)/2);
      RDmaxtmp[0] = RDmintmp[0] + span - 1;
    }
  else
    {
      RDmintmp[0] = -(span/2);
      RDmaxtmp[0] = RDmintmp[0] + span;
    }

  int seg_num =0;
  while (RDmaxtmp[seg_num] < max_delta)
  {
    seg_num++;
    RDmintmp[seg_num] = RDmaxtmp[seg_num-1] + 1;
    RDmaxtmp[seg_num] = RDmintmp[seg_num] + span - 1;
  }
  // check if we went one too far
  if (RDmaxtmp[seg_num] > max_delta)
  {
      RDmintmp[seg_num] = max_delta;
      RDmaxtmp[seg_num] = max_delta;
  }

  const int max_seg_num = seg_num;
  
  VectorWithOffset<int> num_axial_pos_per_segment(-max_seg_num,max_seg_num);
  VectorWithOffset<int> min_ring_difference(-max_seg_num,max_seg_num);
  VectorWithOffset<int> max_ring_difference(-max_seg_num,max_seg_num);
  
  min_ring_difference[0] = RDmintmp[0];
  max_ring_difference[0]= RDmaxtmp[0];
  
  for(int i=1; i<=max_seg_num; i++)
  {
    // KT 28/06/2001 make sure max_ring_diff>min_ring_diff for negative segments
    max_ring_difference[-i]= -RDmintmp[i];
    max_ring_difference[i]= RDmaxtmp[i];
    min_ring_difference[-i]= -RDmaxtmp[i];
    min_ring_difference[i]= RDmintmp[i];    
  }
  
  if (span==1)
  {
    num_axial_pos_per_segment[0] = num_ring;
    for(int i=1; i<=max_seg_num; i++)
    {
      num_axial_pos_per_segment[i]= 
        num_axial_pos_per_segment[-i] = num_ring -i;
    }
  }
  else
  {
    num_axial_pos_per_segment[0] = 2*num_ring -1 ; 
    for( int i=1; i<=max_seg_num; i++)
    {
      num_axial_pos_per_segment[i] = 
        num_axial_pos_per_segment[-i] = (2*num_ring -1 - 2*RDmintmp[i]); 
    }    
  }
  
  const float bin_size = scanner->get_default_bin_size();
  
  
  if (arc_corrected)
    return
    new ProjDataInfoCylindricalArcCorr(scanner,bin_size,
                                       num_axial_pos_per_segment,
                                       min_ring_difference, 
                                       max_ring_difference,
                                       num_views,num_tangential_poss);
  else
    return
    new ProjDataInfoCylindricalNoArcCorr(scanner,
                                       num_axial_pos_per_segment,
                                       min_ring_difference, 
                                       max_ring_difference,
                                       num_views,num_tangential_poss);

}

unique_ptr<ProjDataInfo>
ProjDataInfo::construct_proj_data_info(const shared_ptr<Scanner>& scanner_sptr,
	const int span, const int max_delta,
	const int num_views, const int num_tangential_poss,
	const bool arc_corrected)
{
  unique_ptr<ProjDataInfo> pdi(ProjDataInfoCTI(scanner_sptr, span, max_delta, num_views, num_tangential_poss, arc_corrected));
  return pdi;
}

// KT 28/06/2001 added arc_corrected flag
ProjDataInfo*
ProjDataInfo::ProjDataInfoGE(const shared_ptr<Scanner>& scanner,
			     const int max_delta,
			     const int num_views, const int num_tangential_poss,
                             const bool arc_corrected)
	       
	       
{
  /* mixed span case:
     segment 0 has ring diff -1,0,1,
     other segments have no axial compression
  */
  const int num_rings = scanner->get_num_rings();
  const float bin_size = scanner->get_default_bin_size();
  
  if(max_delta<1)
    error("ProjDataInfo::ProjDataInfoGE: can only handle max_delta>=1\n");

  const int max_segment_num = 
    max_delta==0? 0 : max_delta-1;
  
  VectorWithOffset<int> num_axial_pos_per_segment(-max_segment_num,max_segment_num);
  
  VectorWithOffset<int> min_ring_difference(-max_segment_num,max_segment_num); 
  VectorWithOffset<int> max_ring_difference(-max_segment_num,max_segment_num);
  
  num_axial_pos_per_segment[0] = 2*num_rings-1;
  min_ring_difference[0] = -1; 
  max_ring_difference[0] = 1;
  
  for (int i=1; i<=max_segment_num; i++)
    
  { 
    num_axial_pos_per_segment[i]=
      num_axial_pos_per_segment[-i]=num_rings-i-1;
    
    max_ring_difference[i]= 
      min_ring_difference[i]= i+1;
    max_ring_difference[-i]= 
      min_ring_difference[-i]= -i-1;
    
  }
  
  
  
  if (arc_corrected)
    return
    new ProjDataInfoCylindricalArcCorr(scanner,bin_size,
                                       num_axial_pos_per_segment,
                                       min_ring_difference, 
                                       max_ring_difference,
                                       num_views,num_tangential_poss);
  else
    return
    new ProjDataInfoCylindricalNoArcCorr(scanner,
                                       num_axial_pos_per_segment,
                                       min_ring_difference, 
                                       max_ring_difference,
                                       num_views,num_tangential_poss);
}


ProjDataInfo* ProjDataInfo::ask_parameters()
{

  shared_ptr<Scanner> scanner_ptr(Scanner::ask_parameters());
  
  const bool is_Advance =
    scanner_ptr->get_type() == Scanner::Advance ||
    scanner_ptr->get_type() == Scanner::DiscoveryLS; 
  const bool is_DiscoveryST =
    scanner_ptr->get_type() == Scanner::DiscoveryST; 
  const bool is_GE =
    is_Advance || is_DiscoveryST;

   const int num_views = scanner_ptr->get_max_num_views()/
     ask_num("Mash factor for views",1, scanner_ptr->get_max_num_views(),1);

  const bool arc_corrected =
    ask("Is the data arc-corrected?",true);

   const int num_tangential_poss=
     ask_num("Number of tangential positions",1,
	     scanner_ptr->get_max_num_non_arccorrected_bins(),
	     arc_corrected 
	     ? scanner_ptr->get_default_num_arccorrected_bins()
	     : scanner_ptr->get_max_num_non_arccorrected_bins());
  
   int span = is_GE ? 0 : 1;
   span = 
     ask_num("Span value (use 0 for mixed-span case of the Advance) : ", 0,scanner_ptr->get_num_rings()-1,span);

  
   const int max_delta = ask_num("Max. ring difference acquired : ",
    0,
    scanner_ptr->get_num_rings()-1,
    is_Advance ? 11 : scanner_ptr->get_num_rings()-1);
 

  ProjDataInfo * pdi_ptr =
    span==0
    ? ProjDataInfoGE(scanner_ptr,max_delta,num_views,num_tangential_poss,arc_corrected)
    : ProjDataInfoCTI(scanner_ptr,span,max_delta,num_views,num_tangential_poss,arc_corrected);

  cout << pdi_ptr->parameter_info() <<endl;

  return pdi_ptr;
    
}

/*! Default implementation checks common variables. Needs to be overloaded.
 */
bool
ProjDataInfo::
blindly_equals(const root_type * const that) const
{ 
  const root_type & proj = *that;

  return
   (get_min_view_num()==proj.get_min_view_num()) &&
   (get_max_view_num()==proj.get_max_view_num()) &&
   (get_min_tangential_pos_num() ==proj.get_min_tangential_pos_num())&&
   (get_max_tangential_pos_num() ==proj.get_max_tangential_pos_num())&&
   equal(min_axial_pos_per_seg.begin(), min_axial_pos_per_seg.end(), proj.min_axial_pos_per_seg.begin())&&
   equal(max_axial_pos_per_seg.begin(), max_axial_pos_per_seg.end(), proj.max_axial_pos_per_seg.begin())&&
   (*get_scanner_ptr()== *(proj.get_scanner_ptr()));
}

bool
ProjDataInfo::
operator ==(const root_type& that) const
{ 
  return
    typeid(*this) == typeid(that) &&
    (this == &that ||
     this->blindly_equals(&that)
     );
}

bool
ProjDataInfo::
operator !=(const root_type& that) const
{ 
  return !((*this) == that);
}

/*!
  \return
     \c true only if the types are the same, they are equal, or the range for the
     segments, axial and tangential positions is at least as large.

  \warning Currently view ranges have to be identical.
*/
bool
ProjDataInfo::
operator>=(const ProjDataInfo& proj_data_info) const
{
  if (typeid(*this) != typeid(proj_data_info))
    return false;

  const ProjDataInfo& larger_proj_data_info = *this;

  if (larger_proj_data_info == proj_data_info)
    return true;

  if (proj_data_info.get_max_segment_num() > larger_proj_data_info.get_max_segment_num() ||
      proj_data_info.get_min_segment_num() < larger_proj_data_info.get_min_segment_num() ||
      proj_data_info.get_max_tangential_pos_num() > larger_proj_data_info.get_max_tangential_pos_num() ||
      proj_data_info.get_min_tangential_pos_num() < larger_proj_data_info.get_min_tangential_pos_num())
    return false;

  for (int segment_num=proj_data_info.get_min_segment_num();
       segment_num<=proj_data_info.get_max_segment_num();
       ++segment_num)
    {
      if (proj_data_info.get_max_axial_pos_num(segment_num) > larger_proj_data_info.get_max_axial_pos_num(segment_num) ||
          proj_data_info.get_min_axial_pos_num(segment_num) < larger_proj_data_info.get_min_axial_pos_num(segment_num))
        return false;
    }

  // now check all the rest. That's a bit hard, so what we'll do is reduce the sizes of the larger one
  // to the ones from proj_data_info (which we can safely do as we've checked that they're smaller)
  // and then check for equality.
  // This will check stuff like scanners etc etc...
  shared_ptr<ProjDataInfo> smaller_proj_data_info_sptr(larger_proj_data_info.clone());
  smaller_proj_data_info_sptr->reduce_segment_range(proj_data_info.get_min_segment_num(), proj_data_info.get_max_segment_num());  
  smaller_proj_data_info_sptr->set_min_tangential_pos_num(proj_data_info.get_min_tangential_pos_num());
  smaller_proj_data_info_sptr->set_max_tangential_pos_num(proj_data_info.get_max_tangential_pos_num());

  for (int segment_num=proj_data_info.get_min_segment_num();
       segment_num<=proj_data_info.get_max_segment_num();
       ++segment_num)
    {
      smaller_proj_data_info_sptr->set_min_axial_pos_num(proj_data_info.get_min_axial_pos_num(segment_num), segment_num);
      smaller_proj_data_info_sptr->set_max_axial_pos_num(proj_data_info.get_max_axial_pos_num(segment_num), segment_num);
    }

  return (proj_data_info == *smaller_proj_data_info_sptr);
}

END_NAMESPACE_STIR

