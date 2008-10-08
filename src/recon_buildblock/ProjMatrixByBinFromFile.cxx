//
// $Id$
//
/*
    Copyright (C) 2004- $Date$, Hammersmith Imanet Ltd
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
  \ingroup projection

  \brief Implementation of class stir::ProjMatrixByBinFromFile

  \author Kris Thielemans
  
  $Date$
  $Revision$
*/

#include "stir/ProjDataInterfile.h"
#include "stir/recon_buildblock/ProjMatrixByBinFromFile.h"
#include "stir/recon_buildblock/DataSymmetriesForBins_PET_CartesianGrid.h"
#include "stir/KeyParser.h"
#include "stir/IO/OutputFileFormat.h"
//#include "stir/IO/read_from_file.h"
#include "stir/ProjDataInfo.h"
#include "stir/utilities.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/Succeeded.h"
#include "stir/is_null_ptr.h"
#include "stir/Coordinate3D.h"
#include "stir/info.h"
#include "boost/scoped_ptr.hpp"
#include <fstream>
#include <algorithm>


START_NAMESPACE_STIR


const char * const 
ProjMatrixByBinFromFile::registered_name =
"From File";

ProjMatrixByBinFromFile::
ProjMatrixByBinFromFile()
{
  set_defaults();
}

void 
ProjMatrixByBinFromFile::
initialise_keymap()
{
  parser.add_start_key("Projection Matrix By Bin From File Parameters");
  ProjMatrixByBin::initialise_keymap();

  parser.add_key("template_density_filename", &template_density_filename);
  parser.add_key("template_proj_data_filename", &template_proj_data_filename);
  parser.add_key("data_filename", &data_filename);

  parser.add_key("Version", &this->parsed_version);
  parser.add_key("symmetries type", &this->symmetries_type) ;
    
  //parser.add_key("PET_CartesianGrid symmetries parameters",
  //		 KeyArgument::NONE,	&KeyParser::do_nothing);
  parser.add_key("do_symmetry_90degrees_min_phi", &do_symmetry_90degrees_min_phi);
  parser.add_key("do_symmetry_180degrees_min_phi", &do_symmetry_180degrees_min_phi);
  parser.add_key("do_symmetry_swap_segment", &do_symmetry_swap_segment);
  parser.add_key("do_symmetry_swap_s", &do_symmetry_swap_s);
  parser.add_key("do_symmetry_shift_z", &do_symmetry_shift_z);
  //parser.add_key("End PET_CartesianGrid symmetries parameters",
  //		 KeyArgument::NONE,	&KeyParser::do_nothing);
  parser.add_stop_key("End Projection Matrix By Bin From File Parameters");
}


void
ProjMatrixByBinFromFile::set_defaults()
{
  ProjMatrixByBin::set_defaults();
  template_density_filename="";
  template_proj_data_filename="";
  data_filename="";

  do_symmetry_90degrees_min_phi = true;
  do_symmetry_180degrees_min_phi = true;
  do_symmetry_swap_segment = true;
  do_symmetry_swap_s = true;
  do_symmetry_shift_z = true;
}


bool
ProjMatrixByBinFromFile::post_processing()
{
  if (ProjMatrixByBin::post_processing() == true)
    return true;

  if (this->parsed_version != "1.0")
    { 
      warning("version has to be 1.0");
      return true;
    }
  if (this->symmetries_type != "PET_CartesianGrid")
    { 
      warning("symmetries type has to be PET_CartesianGrid");
      return true;
    }

  if (template_density_filename.size()==0)
    {
      warning("template_density_filename has to be specified.\n");
      return true;
    }
  if (template_proj_data_filename.size()==0)
    {
      warning("template_proj_data_filename has to be specified.\n");
      return true;
    }
  if (data_filename.size()==0)
    {
      warning("data_filename has to be specified.\n");
      return true;
    }
  {
    shared_ptr<ProjData> proj_data_sptr = 
      ProjData::read_from_file(template_proj_data_filename);
    proj_data_info_ptr = 
      proj_data_sptr->get_proj_data_info_ptr()->clone();
  }
  shared_ptr<DiscretisedDensity<3,float> > density_info_sptr =
    DiscretisedDensity<3,float>::read_from_file(template_density_filename);
  {
    const VoxelsOnCartesianGrid<float> * image_info_ptr =
      dynamic_cast<const VoxelsOnCartesianGrid<float>*> (density_info_sptr.get());

    if (image_info_ptr == NULL)
      error("ProjMatrixByBinFromFile initialised with a wrong type of DiscretisedDensity");
 
    densel_range = image_info_ptr->get_index_range();
    voxel_size = image_info_ptr->get_voxel_size();
    origin = image_info_ptr->get_origin();
  }



  symmetries_ptr = 
    new DataSymmetriesForBins_PET_CartesianGrid(proj_data_info_ptr,
                                                density_info_sptr,
                                                do_symmetry_90degrees_min_phi,
                                                do_symmetry_180degrees_min_phi,
						do_symmetry_swap_segment,
						do_symmetry_swap_s,
						do_symmetry_shift_z);

  return false;
}


void
ProjMatrixByBinFromFile::
set_up(		 
    const shared_ptr<ProjDataInfo>& proj_data_info_ptr_v,
    const shared_ptr<DiscretisedDensity<3,float> >& density_info_ptr // TODO should be Info only
    )
{

  const VoxelsOnCartesianGrid<float> * image_info_ptr =
    dynamic_cast<const VoxelsOnCartesianGrid<float>*> (density_info_ptr.get());

  if (image_info_ptr == NULL)
    error("ProjMatrixByBinFromFile set-up with a wrong type of DiscretisedDensity\n");

  // TODO allow for smaller range
  if (densel_range != image_info_ptr->get_index_range())
    error("ProjMatrixByBinFromFile set-up with image with wrong index range\n");
  if (voxel_size != image_info_ptr->get_voxel_size())
    error("ProjMatrixByBinFromFile set-up with image with wrong voxel size\n");
  if (origin != image_info_ptr->get_origin())
    error("ProjMatrixByBinFromFile set-up with image with wrong origin\n");

  /* do consistency checks on projection data.
     It's safe as long as the stored range is larger than what we need.
  */
  {
    boost::scoped_ptr<ProjDataInfo> smaller_proj_data_info_sptr(this->proj_data_info_ptr->clone());
    // first reduce to input segment-range
    {
      const int new_max = std::min(proj_data_info_ptr_v->get_max_segment_num(),
				   this->proj_data_info_ptr->get_max_segment_num());
      const int new_min = std::max(proj_data_info_ptr_v->get_min_segment_num(),
				   this->proj_data_info_ptr->get_min_segment_num());
      smaller_proj_data_info_sptr->reduce_segment_range(new_min, new_max);
    }
    // same for tangential_pos range
    {
      const int new_max = std::min(proj_data_info_ptr_v->get_max_tangential_pos_num(),
				   this->proj_data_info_ptr->get_max_tangential_pos_num());
      const int new_min = std::max(proj_data_info_ptr_v->get_min_tangential_pos_num(),
				   this->proj_data_info_ptr->get_min_tangential_pos_num());
      smaller_proj_data_info_sptr->set_min_tangential_pos_num(new_min);
      smaller_proj_data_info_sptr->set_max_tangential_pos_num(new_max);
    }

    if (*proj_data_info_ptr_v != *smaller_proj_data_info_sptr)
      error("ProjMatrixByBinFromFile set-up with proj data with wrong characteristics");
  }

  if (read_data() ==Succeeded::no)
    error("Exiting\n");
}

Succeeded
ProjMatrixByBinFromFile::
write (std::ostream&fst, const ProjMatrixElemsForOneBin& lor) 
{  
  const Bin bin = lor.get_bin();
  {
    int c;
    c = bin.segment_num(); fst.write ( (char*)&c, sizeof(int));
    c = bin.view_num(); fst.write ( (char*)&c, sizeof(int));
    c = bin.axial_pos_num(); fst.write ( (char*)&c, sizeof(int));
    c = bin.tangential_pos_num(); fst.write ( (char*)&c, sizeof(int));
  }
  {
    std::size_t c= lor.size();
    fst.write( (char*)&c , sizeof(std::size_t));  
  }
  if (!fst)
    return Succeeded::no;
  ProjMatrixElemsForOneBin::const_iterator element_ptr = lor.begin();
  // todo add compression in this loop 
  while (element_ptr != lor.end())
    {           
      short c;
      c = static_cast<short>(element_ptr->coord1());
      fst.write ( (char*)&c, sizeof(short));
      c = static_cast<short>(element_ptr->coord2());
      fst.write ( (char*)&c, sizeof(short));
      c = static_cast<short>(element_ptr->coord3());
      fst.write ( (char*)&c, sizeof(short));
      const float value = element_ptr->get_value();
      fst.write ( (char*)&value, sizeof(float));
      if (!fst)
	return Succeeded::no;

      ++element_ptr;
    } 
  return Succeeded::yes;
} 

//! Read probabilities from stream
Succeeded
ProjMatrixByBinFromFile::
read(std::istream&fst, ProjMatrixElemsForOneBin& lor )
{   
  lor.erase();

  {
    Bin bin;
    int c;
    fst.read( (char*)&c, sizeof(int)); bin.segment_num()=c;
    fst.read( (char*)&c, sizeof(int)); bin.view_num()=c;
    fst.read( (char*)&c, sizeof(int)); bin.axial_pos_num()=c;
    fst.read( (char*)&c, sizeof(int)); bin.tangential_pos_num()=c;
    bin.set_bin_value(0);
    if (bin != lor.get_bin())
      {
	warning("Read bin in wrong order.\nRequired:(s:%d,a:%d,v:%d,t:%d). Read:(s:%d,a:%d,v:%d,t:%d)",
		bin.segment_num(),bin.axial_pos_num(),bin.view_num(),bin.tangential_pos_num(),
		lor.get_bin().segment_num(),lor.get_bin().axial_pos_num(),lor.get_bin().view_num(),lor.get_bin().tangential_pos_num());
	return Succeeded::no;
      }
  }
  std::size_t count;
  fst.read ( (char*)&count, sizeof(std::size_t));

  if (!fst)
    return Succeeded::no;

  lor.reserve(count);

#if 0
  std::size_t buffer_size=count*(sizeof(short)*3+sizeof(float));
  char buffer[buffer_size];
  fst.read(buffer, buffer_size);
  if (!fst)
    return Succeeded::no;
  // todo handle any compression

  std::size_t offset=std::size_t(0);
#endif

  for ( std::size_t i=0; i < count; ++i) 
    { 
      short c1,c2,c3;
#if 1
      fst.read ( (char*)&c1, sizeof(short));
      fst.read ( (char*)&c2, sizeof(short));
      fst.read ( (char*)&c3, sizeof(short));
      float value;
      fst.read ( (char*)&value, sizeof(float));

      if (!fst)
	return Succeeded::no;
#else
      memcpy(buffer + offset, (char*)&c1, sizeof(short));
      offset += sizeof(short);
      memcpy(buffer + offset, (char*)&c2, sizeof(short));
      offset += sizeof(short);
      memcpy(buffer + offset, (char*)&c3, sizeof(short));
      offset += sizeof(short);
      float value;
      memcpy(buffer + offset, (char*)&value, sizeof(float));
      offset += sizeof(float);
#endif
      const ProjMatrixElemsForOneBin::value_type 
	elem(Coordinate3D<int>(c1,c2,c3), value);      
      lor.push_back( elem);		
    }  
  return Succeeded::yes;
}
    
Succeeded
ProjMatrixByBinFromFile::
write_to_file(const string& output_filename_prefix, 
	      const ProjMatrixByBin& proj_matrix,
	      const shared_ptr<ProjDataInfo>& proj_data_info_sptr,
	      const DiscretisedDensity<3,float>& template_density)
{

  string template_density_filename =
    output_filename_prefix + "_template_density";
  {
    if (OutputFileFormat<DiscretisedDensity<3,float> >::default_sptr()->
	write_to_file(template_density_filename,
				    template_density) != Succeeded::yes)
      {
	warning("Error writing template image");
	return Succeeded::no;
      }
  }
  string template_proj_data_filename =
    output_filename_prefix + "_template_proj_data";
  {
    // the following constructor will write an interfile header (and empty data) to disk
    ProjDataInterfile template_projdata(proj_data_info_sptr,
					template_proj_data_filename);
  }

  string header_filename = output_filename_prefix;
  replace_extension(header_filename, ".hpm");
  string data_filename = output_filename_prefix;
  add_extension(data_filename, ".pm");

  {
    std::ofstream header(header_filename.c_str());
    if (!header)
      {
	warning("Error opening header %s",
		header_filename.c_str());
	return Succeeded::no;
      }

    header << "Projection Matrix By Bin From File Parameters:=\n"
	   << "Version := 1.0\n";
    // TODO symmetries should not be hard-coded
    const DataSymmetriesForBins_PET_CartesianGrid& symmetries =
      dynamic_cast<const DataSymmetriesForBins_PET_CartesianGrid&>
      (*proj_matrix.get_symmetries_ptr());
    header << "symmetries type := PET_CartesianGrid\n"
	   << " PET_CartesianGrid symmetries parameters:=\n"
	   << "  do_symmetry_90degrees_min_phi:= " << (symmetries.using_symmetry_90degrees_min_phi() ? 1 : 0) << '\n'
	   << "  do_symmetry_180degrees_min_phi:= " << (symmetries.using_symmetry_180degrees_min_phi() ? 1 : 0) << '\n'
	   << "  do_symmetry_swap_segment:= " << (symmetries.using_symmetry_swap_segment() ? 1 : 0) << '\n'
	   << "  do_symmetry_swap_s:= " << (symmetries.using_symmetry_swap_s() ? 1 : 0) << '\n'
	   << "  do_symmetry_shift_z:= " << (symmetries.using_symmetry_shift_z() ? 1 : 0) << '\n'
	   << " End PET_CartesianGrid symmetries parameters:=\n";

    header << "template proj data filename:=" << template_proj_data_filename << '\n';
    header << "template density filename:=" << template_density_filename << '\n';

    header << "data_filename:=" << data_filename << '\n';

    header << "End Projection Matrix By Bin From File Parameters:=";
  }

  std::ofstream fst;
  open_write_binary(fst, data_filename.c_str());
  
  // loop over bins
  // the complication here is that we cannot just test if each bin in the range is 'basic'
  // and write only those. The reason is that symmetry operations can construct a
  // 'basic' bin outside of the input range (e.g. for tangential_pos_num ranging from -128 to 127).
  // So, we can only loop over all bins, convert to basic bins, and write those.
  // The complication is then that we need to keep track which one we wrote already.
  // Originally, I did this via a std::list<Bin>. Checking if a bin was already written
  // is terribly slow however. Instead, I currently use a vector of shared_ptrs.
  // This wastes only a little bit of memory, but the bounds are difficult to 
  // determine in general.
  // A better approach (and simpler) would be to have access to the internal cache of the 
  // projection matrix.
  {
    // defined here to avoid reallocation for every bin
    ProjMatrixElemsForOneBin lor;

#if 0
    std::list<Bin> already_processed;
#else
    typedef VectorWithOffset<bool> tpos_t;
    typedef VectorWithOffset<shared_ptr<tpos_t> > vpos_t;
    typedef VectorWithOffset<shared_ptr<vpos_t> > apos_t;
    typedef VectorWithOffset<shared_ptr<apos_t> > spos_t;

    // vector that will contain (vectors of bools) to check if we wrote a bin already or not
    // upper boundary takes into account that symmetries convert negative segment_num to positive
    spos_t already_processed(proj_data_info_sptr->get_min_segment_num(), 
			     std::max(proj_data_info_sptr->get_max_segment_num(),
				      -proj_data_info_sptr->get_min_segment_num())); 
#endif
    for (int segment_num = proj_data_info_sptr->get_min_segment_num(); 
	 segment_num <= proj_data_info_sptr->get_max_segment_num();
	 ++segment_num)
    for (int axial_pos_num = proj_data_info_sptr->get_min_axial_pos_num(segment_num);
         axial_pos_num <= proj_data_info_sptr->get_max_axial_pos_num(segment_num);
         ++axial_pos_num)
      for (int view_num = proj_data_info_sptr->get_min_view_num();
	   view_num <= proj_data_info_sptr->get_max_view_num();
	   ++view_num)
	for (int tang_pos_num = proj_data_info_sptr->get_min_tangential_pos_num();
	     tang_pos_num <= proj_data_info_sptr->get_max_tangential_pos_num();
	     ++tang_pos_num)
	  {
	    Bin  bin(segment_num,view_num, axial_pos_num, tang_pos_num);
	    proj_matrix.get_symmetries_ptr()->find_basic_bin(bin);
#if 0
            if (std::find(already_processed.begin(), already_processed.end(), bin)
		!= already_processed.end())
	      continue;

	    already_processed.push_back(bin);
#else
	    if (is_null_ptr(already_processed[bin.segment_num()]))
	      {
		// range attempts to take into account that symmetries normally bring axial_pos_num back to 0 or 1
		already_processed[bin.segment_num()]=new apos_t(std::min(0,proj_data_info_sptr->get_min_axial_pos_num(bin.segment_num())),
								std::max(1,proj_data_info_sptr->get_max_axial_pos_num(bin.segment_num())));
	      }
	    if (is_null_ptr((*already_processed[bin.segment_num()])[bin.axial_pos_num()]))
	      {
		(*already_processed[bin.segment_num()])[bin.axial_pos_num()]=
		  new vpos_t(proj_data_info_sptr->get_min_view_num(),
			     proj_data_info_sptr->get_max_view_num());
	      }
	    if (is_null_ptr((*(*already_processed[bin.segment_num()])[bin.axial_pos_num()])[bin.view_num()]))
	      {
		// range takes into account that symmetries bring negative tangential_pos_num to positive
		(*(*already_processed[bin.segment_num()])[bin.axial_pos_num()])[bin.view_num()] =
		  new tpos_t(proj_data_info_sptr->get_min_tangential_pos_num(),
			     std::max(proj_data_info_sptr->get_max_tangential_pos_num(),
				      -proj_data_info_sptr->get_min_tangential_pos_num()));
		(*(*already_processed[bin.segment_num()])[bin.axial_pos_num()])[bin.view_num()]->fill(false);
	      }
	    if ((*(*(*already_processed[bin.segment_num()])[bin.axial_pos_num()])[bin.view_num()])[bin.tangential_pos_num()])
	      continue;

	    (*(*(*already_processed[bin.segment_num()])[bin.axial_pos_num()])[bin.view_num()])[bin.tangential_pos_num()]=true;
#endif
	    //if (!proj_matrix.get_symmetries_ptr()->is_basic(bin))
	    //  continue;
	    
	    proj_matrix.get_proj_matrix_elems_for_one_bin(lor,bin);
	    if (write(fst, lor) == Succeeded::no)
	      return Succeeded::no;
	  }
  }
  return Succeeded::yes;
}

Succeeded
ProjMatrixByBinFromFile::
read_data()
{
  std::ifstream fst;
  open_read_binary(fst, data_filename.c_str());
  
  // loop over bins
  // TODO replace by a loop that just reads elements one by one as they come
  {
    // defined here to avoid reallocation for every bin
    ProjMatrixElemsForOneBin lor;
    for (int segment_num = proj_data_info_ptr->get_min_segment_num(); 
	 segment_num <= proj_data_info_ptr->get_max_segment_num();
	 ++segment_num)
    for (int axial_pos_num = proj_data_info_ptr->get_min_axial_pos_num(segment_num);
         axial_pos_num <= proj_data_info_ptr->get_max_axial_pos_num(segment_num);
         ++axial_pos_num)
      for (int view_num = proj_data_info_ptr->get_min_view_num();
	   view_num <= proj_data_info_ptr->get_max_view_num();
	   ++view_num)
	for (int tang_pos_num = proj_data_info_ptr->get_min_tangential_pos_num();
	     tang_pos_num <= proj_data_info_ptr->get_max_tangential_pos_num();
	     ++tang_pos_num)
	  {
	    Bin  bin(segment_num,view_num, axial_pos_num, tang_pos_num);
	    bin.set_bin_value(0);
	    Bin bin_copy=bin;
	    this->get_symmetries_ptr()->find_basic_bin(bin);
#if 0
	    info("org:(s:%d,a:%d,v:%d,t:%d). sym:(s:%d,a:%d,v:%d,t:%d)",
		    bin_copy.segment_num(),bin_copy.axial_pos_num(),bin_copy.view_num(),bin_copy.tangential_pos_num(),
		    bin.segment_num(),bin.axial_pos_num(),bin.view_num(),bin.tangential_pos_num());
#endif		    

 	    lor.set_bin(bin);
	    if (this->get_cached_proj_matrix_elems_for_one_bin(lor) == Succeeded::yes)
	      continue;
	    //if (!get_symmetries_ptr()->is_basic(bin))
	    //  continue;
 	    lor.set_bin(bin);
	    if (read(fst, lor) == Succeeded::no)
	      return Succeeded::no;
	    this->cache_proj_matrix_elems_for_one_bin(lor);
	  }
  }
  return Succeeded::yes;
}


void 
ProjMatrixByBinFromFile::
calculate_proj_matrix_elems_for_one_bin(ProjMatrixElemsForOneBin& 
					) const
{
  error("ProjMatrixByBinFromFile element not found in cache (and hence file)");
}
END_NAMESPACE_STIR

