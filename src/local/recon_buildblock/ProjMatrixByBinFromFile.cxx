//
// $Id$
//

/*!
  \file
  \ingroup utilities

  \brief Implementation of class ProjMatrixByBinFromFile

  \author Kris Thielemans
  
  $Date$
  $Revision$
*/
/*
  Copyright (C) 2004- $Date$, Hammersmith Imanet Ltd
  See STIR/LICENSE.txt for details
*/

#include "stir/ProjDataInterfile.h"
#include "local/stir/recon_buildblock/ProjMatrixByBinFromFile.h"
#include "stir/recon_buildblock/DataSymmetriesForBins_PET_CartesianGrid.h"
#include "stir/KeyParser.h"
#include "stir/IO/DefaultOutputFileFormat.h"
#include "stir/ProjDataFromStream.h"
#include "stir/ProjDataInfo.h"
#include "stir/utilities.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/Succeeded.h"
#include "stir/is_null_ptr.h"
#include "stir/Coordinate3D.h"
#include <fstream>
#include <algorithm>
#include <list>


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
  parser.add_start_key("ProjMatrixByBinFromFile Parameters");
  ProjMatrixByBin::initialise_keymap();

  parser.add_key("template_density_filename", &template_density_filename);
  parser.add_key("template_proj_data_filename", &template_proj_data_filename);
    parser.add_key("data_filename", &data_filename);

  parser.add_key("do_symmetry_90degrees_min_phi", &do_symmetry_90degrees_min_phi);
  parser.add_key("do_symmetry_180degrees_min_phi", &do_symmetry_180degrees_min_phi);
  parser.add_key("do_symmetry_swap_segment", &do_symmetry_swap_segment);
  parser.add_key("do_symmetry_swap_s", &do_symmetry_swap_s);
  parser.add_key("do_symmetry_shift_z", &do_symmetry_shift_z);
  parser.add_stop_key("End ProjMatrixByBinFromFile Parameters");
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
      error("ProjMatrixByBinFromFile initialised with a wrong type of DiscretisedDensity\n");
 
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
 
  if (densel_range != image_info_ptr->get_index_range())
    error("ProjMatrixByBinFromFile set-up with image with wrong index range\n");
  if (voxel_size != image_info_ptr->get_voxel_size())
    error("ProjMatrixByBinFromFile set-up with image with wrong voxel size\n");
  if (origin != image_info_ptr->get_origin())
    error("ProjMatrixByBinFromFile set-up with image with wrong origin\n");

  if (*proj_data_info_ptr_v != *proj_data_info_ptr)
    error("ProjMatrixByBinFromFile set-up with proj data with wrong characteristics\n");

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
	warning("Read bin in wrong order?\n");
	return Succeeded::no;
      }
  }
  std::size_t count;
  fst.read ( (char*)&count, sizeof(std::size_t));

  if (!fst)
    return Succeeded::no;

  lor.reserve(count);

  // todo handle the compression 
  for ( std::size_t i=0; i < count; ++i) 
    { 
      short c1,c2,c3;
      fst.read ( (char*)&c1, sizeof(short));
      fst.read ( (char*)&c2, sizeof(short));
      fst.read ( (char*)&c3, sizeof(short));
      float value;
      fst.read ( (char*)&value, sizeof(float));

      if (!fst)
	return Succeeded::no;

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
    DefaultOutputFileFormat output_format;
    if (output_format.write_to_file(template_density_filename,
				    template_density) != Succeeded::yes)
      {
	warning("Error writing template image\n");
	return Succeeded::no;
      }
  }
  string template_proj_data_filename =
    output_filename_prefix + "_template_proj_data";
  {
    ProjDataInterfile template_projdata(proj_data_info_sptr,
					template_proj_data_filename);
    // TODO ideally ProjDataInterfile would add the extension but it doesn't yet
    add_extension(template_proj_data_filename, ".hs");
  }

  string header_filename = output_filename_prefix;
  add_extension(header_filename, ".hpm");
  string data_filename = output_filename_prefix;
  add_extension(data_filename, ".pm");

  {
    std::ofstream header(header_filename.c_str());
    if (!header)
      {
	warning("Error opening header %s\n",
		header_filename.c_str());
	return Succeeded::no;
      }

    header << "ProjMatrixByBinFromFile Parameters:=\n"
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

    header << "End ProjMatrixByBinFromFile:=";
  }

  std::ofstream fst;
  open_write_binary(fst, data_filename.c_str());
  
  // loop over bins
  {
    // defined here to avoid reallocation for every bin
    ProjMatrixElemsForOneBin lor;

    std::list<Bin> already_processed;

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
	    if (find(already_processed.begin(), already_processed.end(), bin)
		!= already_processed.end())
	      continue;

	    already_processed.push_back(bin);
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
  {
    // defined here to avoid reallocation for every bin
    ProjMatrixElemsForOneBin lor;

    std::list<Bin> already_processed;
    /* no std::list.reserove() obviously
    already_processed.reserve((proj_data_info_ptr->get_num_tangential_poss()/2)*
			      (proj_data_info_ptr->get_num_views()/4)*
			      ((proj_data_info_ptr->get_num_segments()+1)/2));
    */    
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
	    get_symmetries_ptr()->find_basic_bin(bin);
	    if (find(already_processed.begin(), already_processed.end(), bin)
		!= already_processed.end())
	      continue;

	    already_processed.push_back(bin);
	    //if (!get_symmetries_ptr()->is_basic(bin))
	    //  continue;
	    lor.set_bin(bin);
	    if (read(fst, lor) == Succeeded::no)
	      return Succeeded::no;
	    cache_proj_matrix_elems_for_one_bin(lor);
	  }
  }
  return Succeeded::yes;
}


void 
ProjMatrixByBinFromFile::
calculate_proj_matrix_elems_for_one_bin(ProjMatrixElemsForOneBin& 
					) const
{
  error("ProjMatrixByBinFromFile element not found in cache (and hence file)\n");
}
END_NAMESPACE_STIR

