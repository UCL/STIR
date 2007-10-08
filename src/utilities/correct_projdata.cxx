//
// $Id$
//
/*
    Copyright (C) 2000- $Date$, Hammersmith Imanet Ltd
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
  \ingroup utilities

  \brief A utility applying/undoing some corrections on projection data

  This is useful to precorrect projection data. There's also the option to undo
  the correction.

  Here's a sample .par file
\verbatim
correct_projdata Parameters := 
  input file := trues.hs

  ; Current way of specifying time frames, pending modifications to
  ; STIR to read time info from the headers
  ; see class documentation for stir::TimeFrameDefinitions for the format of this file
  ; time frame definition filename :=  frames.fdef

  ; if a frame definition file is specified, you can say that the input data
  ; corresponds to a specific time frame
  ; the number should be between 1 and num_frames and defaults to 1
  ; this is currently only used to pass the relevant time to the normalisation
  ; time frame number := 1
  
  ; output file
  ; for future compatibility, do not use an extension in the name of 
  ; the output file. It will be added automatically
  output filename := precorrected

  ; default value for next is -1, meaning 'all segments'
  ; maximum absolute segment number to process := 
 

  ; use data in the input file, or substitute data with all 1's
  ; (useful to get correction factors only)
  ; default is '1'
  ; use data (1) or set to one (0) := 

  ; precorrect data, or undo precorrection
  ; default is '1'
  ; apply (1) or undo (0) correction := 

  ; parameters specifying correction factors
  ; if no value is given, the corresponding correction will not be performed

  ; random coincidences estimate, subtracted before anything else is done
  ;randoms projdata filename := random.hs
  ; normalisation (or binwise multiplication, so can contain attenuation factors as well)
  Bin Normalisation type := from projdata
    Bin Normalisation From ProjData :=
    normalisation projdata filename:= norm.hs
    End Bin Normalisation From ProjData:=

  ; attenuation image, will be forward projected to get attenuation factors
  ; OBSOLETE
  ;attenuation image filename := attenuation_image.hv
  
  ; forward projector used to estimate attenuation factors, defaults to Ray Tracing
  ; OBSOLETE
  ;forward_projector type := Ray Tracing

  ; scatter term to be subtracted AFTER norm+atten correction
  ; defaults to 0
  ;scatter projdata filename := scatter.hs

  ; to interpolate to uniform sampling in 's', set value to 1
  ; arc correction := 1
END:= 
\endverbatim

Time frame definition is only necessary when the normalisation type uses
this time info for dead-time correction.

\warning arc-correction can currently not be undone.

The following gives a brief explanation of the non-obvious parameters. 

<ul>
<li> use data (1) or set to one (0):<br>
Use the data in the input file, or substitute data with all 1's. This is useful to get correction factors only. Its value defaults to 1.
</li>
<li>
apply (1) or undo (0) correction:<br>
Precorrect data, or undo precorrection. Its value defaults to 1.
</li>
<li>
Bin Normalisation type:<br>
Normalisation (or binwise multiplication, so can contain attenuation factors 
as well). \see stir::BinNormalisation
</li>
<li>
attenuation image filename: obsolete<br>
Specify the attenuation image, which will be forward projected to get 
attenuation factors. Has to be in units cm^-1.
This parameter will be removed. Use instead a "chained" bin normalisation 
with a bin normalisation "from attenuation image" 
\see stir::ChainedBinNormalisation
\see stir::BinNormalisationFromAttenuationImage
</li>
<li>
forward_projector type: obsolete<br>
Forward projector used to estimate attenuation factors, defaults to 
Ray Tracing. \see stir::ForwardProjectorUsingRayTracing
This parameter will be removed.
</li>
</ul>

  \author Kris Thielemans

  $Date$
  $Revision$
*/

#include "stir/utilities.h"
#include "stir/CPUTimer.h"
#include "stir/VoxelsOnCartesianGrid.h"
#include "stir/ProjDataInterfile.h"
#include "stir/RelatedViewgrams.h"
#include "stir/ParsingObject.h"
#include "stir/ArcCorrection.h"
#include "stir/Succeeded.h"
#include "stir/recon_buildblock/TrivialBinNormalisation.h"
#include "stir/recon_buildblock/ChainedBinNormalisation.h"
#include "stir/recon_buildblock/BinNormalisationFromAttenuationImage.h"
#include "stir/TrivialDataSymmetriesForViewSegmentNumbers.h"
#include "stir/ArrayFunction.h"
#include "stir/TimeFrameDefinitions.h"
#ifndef USE_PMRT
#include "stir/recon_buildblock/ForwardProjectorByBinUsingRayTracing.h"
#else
#include "stir/recon_buildblock/ForwardProjectorByBinUsingProjMatrixByBin.h"
#include "stir/recon_buildblock/ProjMatrixByBinUsingRayTracing.h"
#endif
#include "stir/is_null_ptr.h"
#include <string>
#include <iostream> 
#include <fstream>
#include <algorithm>

#ifndef STIR_NO_NAMESPACES
using std::cerr;
using std::endl;
using std::fstream;
using std::ifstream;
using std::cout;
using std::string;
#endif

START_NAMESPACE_STIR



// TODO most of this is identical to ReconstructionParameters, so make a common class
/*! \ingroup utilities
  \brief class to do precorrections

  \todo Preliminary class interface. At some point, this class should move to the
  library, instead of being in correct_projdata.cxx.
*/
class CorrectProjDataApplication : public ParsingObject
{
public:

  CorrectProjDataApplication(const char * const par_filename);

  //! set-up variables before processing
  Succeeded set_up();
  //! do precorrection
  /*! set_up() has to be run first */ 
  Succeeded run() const;
  
  // shared_ptrs such that they clean up automatically at exit
  shared_ptr<ProjData> input_projdata_ptr;
  shared_ptr<ProjData> scatter_projdata_ptr;
  shared_ptr<ProjData> randoms_projdata_ptr;
  shared_ptr<ProjData> output_projdata_ptr;
  shared_ptr<BinNormalisation> normalisation_ptr;
  shared_ptr<DiscretisedDensity<3,float> > attenuation_image_ptr;
  shared_ptr<ForwardProjectorByBin> forward_projector_ptr;
  //! apply_or_undo_correction==true means: apply it
  bool apply_or_undo_correction;
  //! use input data, or replace it with all 1's
  /*! <code>use_data_or_set_to_1 == true</code> means: use the data*/
  bool use_data_or_set_to_1;  
  int max_segment_num_to_process;
  int frame_num;
  TimeFrameDefinitions frame_defs;

  bool do_arc_correction;
private:

  virtual void set_defaults();
  virtual void initialise_keymap();
  virtual bool post_processing();
  string input_filename;
  string output_filename;
  string scatter_projdata_filename;
  string atten_image_filename;
  string norm_filename;  
  string randoms_projdata_filename;  
  string frame_definition_filename;
  
  shared_ptr<ArcCorrection> arc_correction_sptr;

};


Succeeded
CorrectProjDataApplication::
run() const
{
  ProjData& output_projdata = *output_projdata_ptr;
  const ProjData& input_projdata = *input_projdata_ptr;

  const bool do_scatter = !is_null_ptr(scatter_projdata_ptr);
  const bool do_randoms = !is_null_ptr(randoms_projdata_ptr);


  // TODO
  shared_ptr<DataSymmetriesForViewSegmentNumbers> symmetries_ptr = 
    is_null_ptr(forward_projector_ptr) ?
      new TrivialDataSymmetriesForViewSegmentNumbers
    :
      forward_projector_ptr->get_symmetries_used()->clone();

  for (int segment_num = output_projdata.get_min_segment_num(); segment_num <= output_projdata.get_max_segment_num() ; segment_num++)
  {
    cerr<<endl<<"Processing segment # "<<segment_num << "(and any related segments)"<<endl;
    for (int view_num=input_projdata.get_min_view_num(); view_num<=input_projdata.get_max_view_num(); ++view_num)
    {    
      const ViewSegmentNumbers view_seg_nums(view_num,segment_num);
      if (!symmetries_ptr->is_basic(view_seg_nums))
        continue;
      
      // ** first fill in the data **      
      RelatedViewgrams<float> 
        viewgrams = input_projdata.get_empty_related_viewgrams(view_seg_nums,
							       symmetries_ptr);
      if (use_data_or_set_to_1)
      {
        viewgrams += 
          input_projdata.get_related_viewgrams(view_seg_nums,
					       symmetries_ptr);
      }	  
      else
      {
        viewgrams.fill(1.F);
      }
      
      if (do_arc_correction && !apply_or_undo_correction)
	{
	  error("Cannot undo arc-correction yet. Sorry.");
	  // TODO
	  //arc_correction_sptr->undo_arc_correction(output_viewgrams, viewgrams);
	}

      if (do_scatter && !apply_or_undo_correction)
      {
        viewgrams += 
          scatter_projdata_ptr->get_related_viewgrams(view_seg_nums,
	                                              symmetries_ptr);
      }

      if (do_randoms && apply_or_undo_correction)
      {
        viewgrams -= 
          randoms_projdata_ptr->get_related_viewgrams(view_seg_nums,
	                                              symmetries_ptr);
      }
#if 0
      if (frame_num==-1)
      {
	int num_frames = frame_def.get_num_frames();
	for ( int i = 1; i<=num_frames; i++)
	{ 
	  //cerr << "Doing frame  " << i << endl; 
	  const double start_frame = frame_def.get_start_time(i);
	  const double end_frame = frame_def.get_end_time(i);
	  //cerr << "Start time " << start_frame << endl;
	  //cerr << " End time " << end_frame << endl;
	  // ** normalisation **
	  if (apply_or_undo_correction)
	  {
	    normalisation_ptr->apply(viewgrams,start_frame,end_frame);
	  }
	  else
	  {
	    normalisation_ptr->undo(viewgrams,start_frame,end_frame);
	  }
	}
      }



      else
#endif
      {      
	const double start_frame = frame_defs.get_start_time(frame_num);
	const double end_frame = frame_defs.get_end_time(frame_num);
	if (apply_or_undo_correction)
	{
	  normalisation_ptr->apply(viewgrams,start_frame,end_frame);
	}
	else
	{
	  normalisation_ptr->undo(viewgrams,start_frame,end_frame);
	}    
      }
      if (do_scatter && apply_or_undo_correction)
      {
        viewgrams -= 
          scatter_projdata_ptr->get_related_viewgrams(view_seg_nums,
	                                              symmetries_ptr);
      }

      if (do_randoms && !apply_or_undo_correction)
      {
        viewgrams += 
          randoms_projdata_ptr->get_related_viewgrams(view_seg_nums,
	                                              symmetries_ptr);
      }

      if (do_arc_correction && apply_or_undo_correction)
	{
	  viewgrams = arc_correction_sptr->do_arc_correction(viewgrams);
	}

      // output
      {
	// Unfortunately, segment range in output_projdata and input_projdata can be
	// different. 
	// Hence, output_projdata.set_related_viewgrams(viewgrams) would not work.
	// So, we need an extra viewgrams object to take this into account.
	// The trick relies on calling Array::operator+= instead of 
	// RelatedViewgrams::operator=
	RelatedViewgrams<float> 
	  output_viewgrams = 
	  output_projdata.get_empty_related_viewgrams(view_seg_nums,
						    symmetries_ptr);
	  output_viewgrams += viewgrams;

	  if (!(output_projdata.set_related_viewgrams(viewgrams) == Succeeded::yes))
	    {
	      warning("CorrectProjData: Error set_related_viewgrams\n");
	      return Succeeded::no;
	    }
      }
      
    }
        
  }
  return Succeeded::yes;
}    


void 
CorrectProjDataApplication::
set_defaults()
{
  input_projdata_ptr = 0;
  max_segment_num_to_process = -1;
  normalisation_ptr = 0;
  use_data_or_set_to_1= true;
  apply_or_undo_correction = true;
  scatter_projdata_filename = "";
  atten_image_filename = "";
  norm_filename = "";
  normalisation_ptr = new TrivialBinNormalisation;
  randoms_projdata_filename = "";
  attenuation_image_ptr = 0;
  frame_num = 1;
  frame_definition_filename = "";

#ifndef USE_PMRT
  forward_projector_ptr =
    new ForwardProjectorByBinUsingRayTracing;
#else
  shared_ptr<ProjMatrixByBin> PM = 
    new  ProjMatrixByBinUsingRayTracing;
  forward_projector_ptr =
    new ForwardProjectorByBinUsingProjMatrixByBin(PM); 
#endif

  do_arc_correction= false;
}

void 
CorrectProjDataApplication::
initialise_keymap()
{
  parser.add_start_key("correct_projdata Parameters");
  parser.add_key("input file",&input_filename);
  parser.add_key("time frame definition filename", &frame_definition_filename); 
  parser.add_key("time frame number", &frame_num);

  parser.add_key("output filename",&output_filename);
  parser.add_key("maximum absolute segment number to process", &max_segment_num_to_process);
 
  parser.add_key("use data (1) or set to one (0)", &use_data_or_set_to_1);
  parser.add_key("apply (1) or undo (0) correction", &apply_or_undo_correction);
  parser.add_parsing_key("Bin Normalisation type", &normalisation_ptr);
  parser.add_key("randoms projdata filename", &randoms_projdata_filename);
  parser.add_key("attenuation image filename", &atten_image_filename);
  parser.add_parsing_key("forward projector type", &forward_projector_ptr);
  parser.add_key("scatter_projdata_filename", &scatter_projdata_filename);
  parser.add_key("arc correction", &do_arc_correction);
  parser.add_stop_key("END");
}


bool
CorrectProjDataApplication::
post_processing()
{
  if (is_null_ptr(normalisation_ptr))
  {
    warning("Invalid normalisation object\n");
    return true;
  }

  // read time frame def 
   if (frame_definition_filename.size()!=0)
    frame_defs = TimeFrameDefinitions(frame_definition_filename);
   else
    {
      // make a single frame starting from 0 to 1.
      vector<pair<double, double> > frame_times(1, pair<double,double>(0,1));
      frame_defs = TimeFrameDefinitions(frame_times);
    }

  if (frame_num<=0)
    {
      warning("frame_num should be >= 1 \n");
      return true;
    }

  if (static_cast<unsigned>(frame_num)> frame_defs.get_num_frames())
    {
      warning("frame_num is %d, but should be less than the number of frames %d.\n",
	      frame_num, frame_defs.get_num_frames());
      return true;
    }
  input_projdata_ptr = ProjData::read_from_file(input_filename);

  if (scatter_projdata_filename!="" && scatter_projdata_filename != "0")
    scatter_projdata_ptr = ProjData::read_from_file(scatter_projdata_filename);

  if (randoms_projdata_filename!="" && randoms_projdata_filename != "0")
    randoms_projdata_ptr = ProjData::read_from_file(randoms_projdata_filename);

  return false;
}

Succeeded
CorrectProjDataApplication::
set_up()
{
  const int max_segment_num_available =
    input_projdata_ptr->get_max_segment_num();
  if (max_segment_num_to_process<0 ||
      max_segment_num_to_process > max_segment_num_available)
    max_segment_num_to_process = max_segment_num_available;
  shared_ptr<ProjDataInfo>  
    input_proj_data_info_sptr(input_projdata_ptr->get_proj_data_info_ptr()->clone());
  shared_ptr<ProjDataInfo> output_proj_data_info_sptr;

  if (!do_arc_correction)
    output_proj_data_info_sptr = input_proj_data_info_sptr;
  else
    {
      arc_correction_sptr = 
	shared_ptr<ArcCorrection>(new ArcCorrection);
      arc_correction_sptr->set_up(input_proj_data_info_sptr);
      output_proj_data_info_sptr =
	arc_correction_sptr->get_arc_corrected_proj_data_info_sptr();
    }
  output_proj_data_info_sptr->reduce_segment_range(-max_segment_num_to_process, 
					  max_segment_num_to_process);

  // construct output_projdata
  {
#if 0
    // attempt to do mult-frame data, but then we should have different input data anyway
    if (frame_definition_filename.size()!=0 && frame_num==-1)
      {
	const int num_frames = frame_defs.get_num_frames();
	for ( int current_frame = 1; current_frame <= num_frames; current_frame++)
	  {
	    char ext[50];
	    sprintf(ext, "_f%dg1b0d0", current_frame);
	    const string output_filename_with_ext = output_filename + ext;	
	    output_projdata_ptr = new ProjDataInterfile(output_proj_data_info_sptr,output_filename_with_ext);
	  }
      }
    else
#endif
      {
	string output_filename_with_ext = output_filename;
#if 0
	if (frame_definition_filename.size()!=0)
	  {
	    char ext[50];
	    sprintf(ext, "_f%dg1b0d0", frame_num);
	    output_filename_with_ext += ext;
	  }
#endif
      output_projdata_ptr = new ProjDataInterfile(output_proj_data_info_sptr,output_filename_with_ext);
      }

  }
 
  // read attenuation image and add it to the normalisation object
  if(atten_image_filename!="0" && atten_image_filename!="")
    {
      
      shared_ptr<BinNormalisation> atten_sptr
	(new BinNormalisationFromAttenuationImage(atten_image_filename,
						  forward_projector_ptr));
      
      normalisation_ptr = 
	shared_ptr<BinNormalisation>
	( new ChainedBinNormalisation(normalisation_ptr,
				      atten_sptr));
    }
  else
    {
      // get rid of this object for now
      // this is currently checked to find the symmetries: bad
      // TODO
      forward_projector_ptr = 0;
    }

  // set up normalisation object
  if (
      normalisation_ptr->set_up(input_proj_data_info_sptr)
      != Succeeded::yes)
    {
      warning("correct_projdata: set-up of normalisation failed\n");
      return Succeeded::no;
    }

  return Succeeded::yes;
  
}

CorrectProjDataApplication::
CorrectProjDataApplication(const char * const par_filename)
{
  set_defaults();
  if (par_filename!=0)
    parse(par_filename) ;
  else
    ask_parameters();

}

END_NAMESPACE_STIR

USING_NAMESPACE_STIR

int main(int argc, char *argv[])
{
  
  if(argc!=2) 
  {
    cerr<<"Usage: " << argv[0] << " par_file\n"
       	<< endl; 
  }
  CorrectProjDataApplication correct_proj_data_application( argc==2 ? argv[1] : 0);
 
  if (argc!=2)
    {
      cerr << "Corresponding .par file input \n"
	   << correct_proj_data_application.parameter_info() << endl;
    }
    

  CPUTimer timer;
  timer.start();

  if (correct_proj_data_application.set_up() == Succeeded::no)
    return EXIT_FAILURE;

  Succeeded success =
    correct_proj_data_application.run();
  timer.stop();
  cerr << "CPU time : " << timer.value() << "secs" << endl;
  return success==Succeeded::yes ?
    EXIT_SUCCESS : EXIT_FAILURE;

}
