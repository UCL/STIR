//
// $Id$
//
/*!
  \file
  \ingroup projdata  

  \brief Implementations for non-inline functions of class ProjData

  \author Kris Thielemans
  \author PARAPET project

  $Date$
  $Revision$
*/
/*
    Copyright (C) 2000 PARAPET partners
    Copyright (C) 2000- $Date$, Hammersmith Imanet Ltd
    See STIR/LICENSE.txt for details
*/
#include "stir/ProjData.h"
#include "stir/Succeeded.h"
#include "stir/RelatedViewgrams.h"
#include "stir/SegmentBySinogram.h"
#include "stir/SegmentByView.h"
#include "stir/Viewgram.h"
#include "stir/DataSymmetriesForViewSegmentNumbers.h"

#include "stir/utilities.h" // TODO remove (temporary for test GEAdvance)
// for read_from_file
#include "stir/IO/interfile.h"
#include "stir/ProjDataFromStream.h" // needed for converting ProjDataFromStream* to ProjData*
#ifndef STIR_DEVEL
#include "stir/ProjDataGEAdvance.h"
#else
#include "stir/IO/ProjDataVOLPET.h"
#endif
#include "stir/IO/stir_ecat7.h"
#include "stir/ViewSegmentNumbers.h"
#include "stir/is_null_ptr.h"
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
   <li> GE VOLPET data (via class ProjDataVOLPET)
   <li> Interfile (using  read_interfile_PDFS())
   <li> ECAT 7 3D sinograms and attenuation files 
   </ul>

   Developer's note: ideally the return value would be an auto_ptr.
*/
shared_ptr<ProjData> 
ProjData::
read_from_file(const string& filename,
	       const ios::openmode openmode)
{
  fstream * input = new fstream(filename.c_str(), openmode | ios::binary);
  if (! *input)
    error("ProjData::read_from_file: error opening file %s", filename.c_str());

  const int max_length=300;
  char signature[max_length];
  input->read(signature, max_length);
  signature[max_length-1]='\0';

  // GE Advance
  if (strncmp(signature, "2D3D", 4) == 0)
  {
//    if (ask("Read with old code (Y) or new (N)?",false))
#ifndef STIR_DEVEL 
      {
#ifndef NDEBUG
	warning("ProjData::read_from_file trying to read %s as GE Advance file", 
		filename.c_str());
#endif
	return shared_ptr<ProjData>( new ProjDataGEAdvance(input) );
      }
      //else
#else // use VOLPET
      {
#ifndef NDEBUG
	warning("ProjData::read_from_file trying to read %s as GE VOLPET file", 
		filename.c_str());
#endif
	delete input;// TODO no longer use pointer after getting rid of ProjDataGEAdvance
	return shared_ptr<ProjData>( new ProjDataVOLPET(filename) );
      }
  }
#endif // STIR_DEVEL to differentiate between Advance and VOLPET code

  delete input;// TODO no longer use pointer after getting rid of ProjDataGEAdvance

#ifdef HAVE_LLN_MATRIX
  // ECAT 7
  if (strncmp(signature, "MATRIX", 6) == 0)
  {
#ifndef NDEBUG
    warning("ProjData::read_from_file trying to read %s as ECAT7", filename.c_str());
#endif
    USING_NAMESPACE_ECAT;
    USING_NAMESPACE_ECAT7;

    if (is_ECAT7_emission_file(filename) || is_ECAT7_attenuation_file(filename))
    {
      warning("\nReading frame 1, gate 1, data 0, bed 0 from file %s",
	      filename.c_str());
      return ECAT7_to_PDFS(filename, /*frame_num, gate_num, data_num, bed_num*/1,1,0,0);
    }
    else
    {
      if (is_ECAT7_file(filename))
	warning("ProjData::read_from_file ECAT7 file %s is of unsupported file type", filename.c_str());
    }
  }
#endif // HAVE_LLN_MATRIX

  // Interfile
  if (is_interfile_signature(signature))
  {
#ifndef NDEBUG
    warning("ProjData::read_from_file trying to read %s as Interfile", filename.c_str());
#endif
    ProjData * ptr =
      read_interfile_PDFS(filename, openmode);
    if (!is_null_ptr(ptr))
      return ptr;
  }


  error("\nProjData::read_from_file could not read projection data %s.\n"
	"Unsupported file format? Aborting.",
	  filename.c_str());
  return 0;
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
