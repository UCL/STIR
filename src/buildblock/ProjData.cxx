//
// $Id$
//
/*!
  \file
  \ingroup buildblock  

  \brief Implementations for non-inline functions of class ProjData

  \author Kris Thielemans
  \author PARAPET project

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, IRSL
    See STIR/LICENSE.txt for details
*/
#include "stir/ProjData.h"
#include "stir/Succeeded.h"
#include "stir/RelatedViewgrams.h"
#include "stir/SegmentBySinogram.h"
#include "stir/SegmentByView.h"
#include "stir/Viewgram.h"
#include "stir/DataSymmetriesForViewSegmentNumbers.h"

// for read_from_file
#include "stir/interfile.h"
#include "stir/ProjDataFromStream.h"
#include "stir/ProjDataGEAdvance.h"
#include "stir/ViewSegmentNumbers.h"
#include <cstring>
#include <fstream>

#ifndef STIR_NO_NAMESPACES
using std::istream;
using std::fstream;
using std::ios;
#endif

START_NAMESPACE_STIR

/*! 
   This function will attempt to determine the type of projection data in the file,
   construct an object of the appropriate type, and return a pointer to 
   the object.

   The return value is a shared_ptr, to make sure that the caller will
   delete the object.

   If more than 1 projection data set is in the file, only the first is read.

   When the file is not readable for some reason, the program is aborted
   by calling error().

   Currently supported:
   <ul>
   <li> ProjDataGEAdvance
   <li> ProjDataFromStream, specified via an Interfile header.
   </ul>

   Developer's note: ideally the return value would be an auto_ptr.
*/
shared_ptr<ProjData> 
ProjData::
read_from_file(const string& filename,
	       const ios::openmode openmode)
{
  iostream * input = new fstream(filename.c_str(), openmode | ios::binary);
  if (! *input)
    error("ProjData::read_from_file: error opening file %s\n", filename.c_str());

  const int max_length=300;
  char signature[max_length];
  input->read(signature, max_length);

  shared_ptr<ProjData> result = 0;

  if (strncmp(signature, "2D3D", 4) == 0)
    result = shared_ptr<ProjData>( new ProjDataGEAdvance(input) );
  else
  {
    delete input;
    result = shared_ptr<ProjData>(read_interfile_PDFS(filename, openmode));
  }

  if (result.use_count() == 0)
    error("\nProjData::read_from_file could not read projection data %s. Aborting.\n",
	  filename.c_str());
  return result;
}


Viewgram<float> 
ProjData::get_empty_viewgram(const int view_num, const int segment_num, 
			     const bool make_num_tangential_poss_odd) const
{
  return
    proj_data_info_ptr->get_empty_viewgram(view_num, segment_num, make_num_tangential_poss_odd);
}

Sinogram<float>
ProjData::get_empty_sinogram(const int ax_pos_num, const int segment_num,
			     const bool make_num_tangential_poss_odd) const
{
  return
    proj_data_info_ptr->get_empty_sinogram(ax_pos_num, segment_num, make_num_tangential_poss_odd);
}


SegmentBySinogram<float>
ProjData::get_empty_segment_by_sinogram(const int segment_num, 
      const bool make_num_tangential_poss_odd) const
{
  return
    proj_data_info_ptr->get_empty_segment_by_sinogram(segment_num, make_num_tangential_poss_odd);
}  


SegmentByView<float>
ProjData::get_empty_segment_by_view(const int segment_num, 
				   const bool make_num_tangential_poss_odd) const
{
  return
    proj_data_info_ptr->get_empty_segment_by_view(segment_num, make_num_tangential_poss_odd);

}

RelatedViewgrams<float> 
ProjData::get_empty_related_viewgrams(const ViewSegmentNumbers& view_segmnet_num,
                   //const int view_num, const int segment_num,
		   const shared_ptr<DataSymmetriesForViewSegmentNumbers>& symmetries_used,
		   const bool make_num_tangential_poss_odd) const
{
  return
    proj_data_info_ptr->get_empty_related_viewgrams(view_segmnet_num, symmetries_used, make_num_tangential_poss_odd);
}


RelatedViewgrams<float> 
ProjData::get_related_viewgrams(const ViewSegmentNumbers& view_segmnet_num,
                   //const int view_num, const int segment_num,
		   const shared_ptr<DataSymmetriesForViewSegmentNumbers>& symmetries_used,
		   const bool make_num_bins_odd) const
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
    viewgrams.push_back(get_viewgram(pairs[i].view_num(),
                                          pairs[i].segment_num(), make_num_bins_odd));
  }

  return RelatedViewgrams<float>(viewgrams, symmetries_used);
}


Succeeded 
ProjData::set_related_viewgrams( const RelatedViewgrams<float>& viewgrams) 
{

  RelatedViewgrams<float>::const_iterator r_viewgrams_iter = viewgrams.begin();
  while( r_viewgrams_iter!=viewgrams.end())
  {
    if (set_viewgram(*r_viewgrams_iter)== Succeeded::no)
      return Succeeded::no;
      ++r_viewgrams_iter;
  }
  return Succeeded::yes;
}

#if 0
  for (int i=0; i<viewgrams.get_num_viewgrams(); ++i)
  {
    if (set_viewgram(viewgrams.get_viewgram_reference(i)) == Succeeded::no)
      return Succeeded::no;
  }
  return Succeeded::yes;
}
#endif

SegmentBySinogram<float> ProjData::get_segment_by_sinogram(const int segment_num) const
{
  SegmentBySinogram<float> segment =
    proj_data_info_ptr->get_empty_segment_by_sinogram(segment_num,false);
  // TODO optimise to get shared proj_data_info_ptr
  for (int view_num = get_min_view_num(); view_num <= get_max_view_num(); ++view_num)
    segment.set_viewgram(get_viewgram(view_num, segment_num, false));

  return segment;
}

SegmentByView<float> ProjData::get_segment_by_view(const int segment_num) const
{
  SegmentByView<float> segment =
    proj_data_info_ptr->get_empty_segment_by_view(segment_num,false);
  // TODO optimise to get shared proj_data_info_ptr
  for (int view_num = get_min_view_num(); view_num <= get_max_view_num(); ++view_num)
    segment.set_viewgram(get_viewgram(view_num, segment_num, false));

  return segment;
}

Succeeded 
ProjData::set_segment(const SegmentBySinogram<float>& segment)
{
  for (int view_num = get_min_view_num(); view_num <= get_max_view_num(); ++view_num)
  {
    if(set_viewgram(segment.get_viewgram(segment.get_segment_num()))
        == Succeeded::no)
	return Succeeded::no;
  }
  return Succeeded::yes;
}

Succeeded 
ProjData::set_segment(const SegmentByView<float>& segment)
{
  for (int view_num = get_min_view_num(); view_num <= get_max_view_num(); ++view_num)
  {
    if(set_viewgram(segment.get_viewgram(segment.get_segment_num()))
        == Succeeded::no)
	return Succeeded::no;
  }
  return Succeeded::yes;
}
  
END_NAMESPACE_STIR
