//
// $Id$
//
/*
    Copyright (C) 2000- $Date$, Hammersmith Imanet Ltd
    See STIR/LICENSE.txt for details
*/
/*!
  \file
  \ingroup utilities

  \brief A utility preparing some projection data for further processing with 
  iterative reconstructions. See stir::PrepareProjData.

  \author Kris Thielemans

  $Date$
  $Revision$
*/

#include "stir/utilities.h"
#include "stir/CPUTimer.h"
#include "stir/ProjDataInterfile.h"
#include "stir/RelatedViewgrams.h"
#include "stir/TrivialDataSymmetriesForViewSegmentNumbers.h"
#include "stir/ParsingObject.h"
#include "stir/is_null_ptr.h"
#include "stir/Succeeded.h"
#include "stir/TimeFrameDefinitions.h"

#include "stir/recon_buildblock/TrivialBinNormalisation.h"

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




/*! \ingroup recon_buildblock
  \brief A preliminary class to prepare files for iterative reconstruction

  The intention of this class is to read measured data (and some processed data)
  and construct from these the projection data that are used in ML/MAP estimates, i.e.
  - normalisation times attenuation file
  - additive terms

  \par Parameter file format
  \verbatim
  Prepare projdata Parameters:=
  ; next defaults to using all segments
  ; maximum absolute segment number to process:= ...
  
  ;;;;;;;;; input, some is optional depending on what you ask for as output 
  prompts_projdata_filename:= ...
  trues_projdata_filename:= ...
  precorrected_projdata_filename:= ...
  randoms_projdata_filename:= ...

  time frame definition filename:= ...
  time frame number:= ...
  ; normalisation (should contain attenuation as well)
  Bin Normalisation type:= ...

  ; if (prompts+randoms or trues) and precorrected data are given
  ;   construct scatter term by subtraction and write to file
  ; else if following is set, use it for scatter term
  ; else scatter term will be set to 0
  scatter_projdata_filename:= ...

  ;;;;;;;; output files

  normatten_projdata_filename:= ...

  Shifted_Poisson_numerator_projdata_filename:= ...
  Shifted_Poisson_denominator_projdata_filename:= ...
  ; name for additive term in denominator of a prompts reconstruction
  prompts_denominator_projdata_filename:= ...
 
  END Prepare projdata Parameters:=
  \endverbatim
*/
class PrepareProjData : public ParsingObject
{
public:

  PrepareProjData(const char * const par_filename);
  void doit();

private:
  // shared_ptr's such that they clean up automatically at exit
  shared_ptr<ProjData> prompts_projdata_ptr;
  shared_ptr<ProjData> trues_projdata_ptr;
  shared_ptr<ProjData> precorrected_projdata_ptr;
  shared_ptr<ProjData> randoms_projdata_ptr;
  shared_ptr<BinNormalisation> normalisation_ptr;
  
  
  shared_ptr<ProjData> normatten_projdata_ptr;
  shared_ptr<ProjData> scatter_projdata_ptr;
  shared_ptr<ProjData> Shifted_Poisson_numerator_projdata_ptr;
  shared_ptr<ProjData> Shifted_Poisson_denominator_projdata_ptr;
  shared_ptr<ProjData> prompts_denominator_projdata_ptr;
  TimeFrameDefinitions frame_defs;
 
  
  int max_segment_num_to_process;
  int current_frame_num;
private:
  bool can_make_trues; // if prompts and randoms are given
  bool do_Shifted_Poisson;
  bool do_prompts; // if we need to construct the additive term in the denominator for a reconstruction of prompts
  bool do_scatter; // if we need to find scatter by subtracting precorrected data from the trues

  // used to create new viewgrams etc
  shared_ptr<ProjDataInfo>  output_data_info_ptr;
  shared_ptr<ProjData> template_projdata_ptr;

  virtual void set_defaults();
  virtual void initialise_keymap();

  string prompts_projdata_filename;
  string trues_projdata_filename;
  string precorrected_projdata_filename;
  string randoms_projdata_filename;
  
  string normatten_projdata_filename;
  string scatter_projdata_filename;
  string Shifted_Poisson_numerator_projdata_filename;
  string Shifted_Poisson_denominator_projdata_filename;
  string prompts_denominator_projdata_filename;
  string frame_definition_filename;
  
};

void 
PrepareProjData::
set_defaults()
{
  prompts_projdata_ptr = 0;
  trues_projdata_ptr = 0;
  precorrected_projdata_ptr = 0;
  randoms_projdata_ptr = 0;
  normalisation_ptr = new TrivialBinNormalisation;
  scatter_projdata_ptr=0;
  max_segment_num_to_process = -1;
  prompts_projdata_filename = "";
  trues_projdata_filename = "";
  precorrected_projdata_filename = "";
  randoms_projdata_filename = "";

  normatten_projdata_filename = "";
  scatter_projdata_filename = "";
  Shifted_Poisson_numerator_projdata_filename = "";
  Shifted_Poisson_denominator_projdata_filename = "";

  current_frame_num = 1;  
}

void 
PrepareProjData::
initialise_keymap()
{
  parser.add_start_key("Prepare projdata Parameters");
  parser.add_parsing_key("Bin Normalisation type", &normalisation_ptr);
  parser.add_key("prompts_projdata_filename", &prompts_projdata_filename);
  parser.add_key("trues_projdata_filename", &trues_projdata_filename);
  parser.add_key("precorrected_projdata_filename", &precorrected_projdata_filename);
  parser.add_key("randoms_projdata_filename", &randoms_projdata_filename);

  parser.add_key("time frame definition filename", &frame_definition_filename); 
  parser.add_key("time frame number", &current_frame_num); 
  
  
  parser.add_key("normatten_projdata_filename", &normatten_projdata_filename);
  parser.add_key("scatter_projdata_filename", &scatter_projdata_filename);
  parser.add_key("Shifted_Poisson_numerator_projdata_filename", &Shifted_Poisson_numerator_projdata_filename);
  parser.add_key("Shifted_Poisson_denominator_projdata_filename", &Shifted_Poisson_denominator_projdata_filename);
  parser.add_key("prompts_denominator_projdata_filename", &prompts_denominator_projdata_filename);
  parser.add_key("maximum absolute segment number to process", &max_segment_num_to_process);
 
  parser.add_stop_key("END Prepare projdata Parameters");
}

PrepareProjData::
PrepareProjData(const char * const par_filename)
{
  set_defaults();
  if (par_filename!=0)
    parse(par_filename) ;
  else
    ask_parameters();

  if (is_null_ptr(normalisation_ptr))
    {
      warning("Invalid normalisation type\n");
      exit(EXIT_FAILURE);
    }
  can_make_trues =
    prompts_projdata_filename.size()!=0 && 
    randoms_projdata_filename.size()!=0;

  do_scatter = 
    (trues_projdata_filename.size()!=0 || can_make_trues) && 
    precorrected_projdata_filename.size()!=0;

  if (prompts_projdata_filename.size()!=0)
    prompts_projdata_ptr = ProjData::read_from_file(prompts_projdata_filename);

  if (trues_projdata_filename.size()!=0)
    trues_projdata_ptr = ProjData::read_from_file(trues_projdata_filename);

  if (precorrected_projdata_filename.size()!=0)
    precorrected_projdata_ptr = ProjData::read_from_file(precorrected_projdata_filename);

  do_Shifted_Poisson = 
    Shifted_Poisson_numerator_projdata_filename.size() != 0;
  if (do_Shifted_Poisson && randoms_projdata_filename.size()==0)
    {
      warning("Shifted Poisson data asked for, but no randoms present\n");
      exit(EXIT_FAILURE);
    }
  if (do_Shifted_Poisson && trues_projdata_filename.size()==0)
    {
      warning("Shifted Poisson data asked for, but no trues present\n");
      exit(EXIT_FAILURE);
    }

  do_prompts =
    prompts_denominator_projdata_filename.size() != 0;

  if (do_prompts && randoms_projdata_filename.size()==0)
    {
      warning("Prompts data asked for, but no randoms present\n");
      exit(EXIT_FAILURE);
    }


  if (do_Shifted_Poisson || do_prompts)
     randoms_projdata_ptr = ProjData::read_from_file(randoms_projdata_filename);

  scatter_projdata_ptr = 
       !do_scatter && scatter_projdata_filename.size()!=0 ?
       ProjData::read_from_file(scatter_projdata_filename) : 0;

  // read time frame def 
  if (frame_definition_filename.size()!=0)
    frame_defs = TimeFrameDefinitions(frame_definition_filename);
  else
    {
      // make a single frame starting from 0 to 1
      warning("No time frame definitions present.\n"
	      "If the normalisation type needs time info for the dead-time correction,\n"
	      "you will get wrong results\n");
      vector<pair<double, double> > frame_times(1, pair<double,double>(0,1));
      frame_defs = TimeFrameDefinitions(frame_times);
    }

  if (current_frame_num < 1 ||
      static_cast<unsigned int>(current_frame_num) > frame_defs.get_num_frames())
    {
      warning("\nFrame number %d is out of range for frame definitions\n", current_frame_num);
      exit(EXIT_FAILURE);
    }

  // construct output projdata
  // and set_up normalisation
  {
    // get output_data_info_ptr from one of the input files

    if (!is_null_ptr(trues_projdata_ptr))
      output_data_info_ptr= 
	trues_projdata_ptr->get_proj_data_info_ptr()->clone();
    else if (!is_null_ptr(randoms_projdata_ptr))
      output_data_info_ptr= 
	randoms_projdata_ptr->get_proj_data_info_ptr()->clone();
    else if (!is_null_ptr(precorrected_projdata_ptr))
      output_data_info_ptr= 
	precorrected_projdata_ptr->get_proj_data_info_ptr()->clone();
    else
      {
	warning("\nAt least one of these input files must be set: trues, randoms, precorrected\n");
	exit(EXIT_FAILURE);
      }

    // set segment range

    const int max_segment_num_available =
      output_data_info_ptr->get_max_segment_num();
    if (max_segment_num_to_process<0 ||
	max_segment_num_to_process > max_segment_num_available)
      max_segment_num_to_process = max_segment_num_available;

    output_data_info_ptr->reduce_segment_range(-max_segment_num_to_process, 
					       max_segment_num_to_process);


    if (normalisation_ptr->set_up(output_data_info_ptr) != Succeeded::yes)
      {
	warning("Error initialisation normalisation\n");
	exit(EXIT_FAILURE);
      }

    // open other files

    if (normatten_projdata_filename.size()!=0)
      normatten_projdata_ptr = 
	new ProjDataInterfile(output_data_info_ptr, normatten_projdata_filename);
    if (do_scatter)
      scatter_projdata_ptr = 
	new ProjDataInterfile(output_data_info_ptr, scatter_projdata_filename);
    if (do_Shifted_Poisson)
    {
      Shifted_Poisson_numerator_projdata_ptr = 
	new ProjDataInterfile(output_data_info_ptr, Shifted_Poisson_numerator_projdata_filename);
      Shifted_Poisson_denominator_projdata_ptr = 
	new ProjDataInterfile(output_data_info_ptr, Shifted_Poisson_denominator_projdata_filename);
    }
    if (do_prompts)
    {
      prompts_denominator_projdata_ptr = 
	new ProjDataInterfile(output_data_info_ptr, prompts_denominator_projdata_filename);
    }
  }

}



void
PrepareProjData::
doit()
{
  shared_ptr<DataSymmetriesForViewSegmentNumbers> symmetries_ptr =
    new TrivialDataSymmetriesForViewSegmentNumbers;

  // take these out of the loop to avoid reallocation (but it's ugly)
  RelatedViewgrams<float> normatten_viewgrams;
  RelatedViewgrams<float> scatter_viewgrams;
  RelatedViewgrams<float> trues_viewgrams;
  RelatedViewgrams<float> randoms_viewgrams;
  RelatedViewgrams<float> Shifted_Poisson_numerator_viewgrams;
  RelatedViewgrams<float> Shifted_Poisson_denominator_viewgrams;
  RelatedViewgrams<float> prompts_denominator_viewgrams;


  for (int segment_num = -max_segment_num_to_process; segment_num <= max_segment_num_to_process; segment_num++)
  {
    cerr<<endl<<"Processing segment #"<<segment_num<<endl;
    for (int view_num=normatten_projdata_ptr->get_min_view_num(); view_num<=normatten_projdata_ptr->get_max_view_num(); ++view_num)
    {    
      const ViewSegmentNumbers view_seg_num(view_num,segment_num);

      if (!symmetries_ptr->is_basic(view_seg_num))
	continue;

      bool already_read_randoms = false;

      // ** first do normalisation (and fill in  normatten) **
      
      /*RelatedViewgrams<float>*/ normatten_viewgrams = 
        output_data_info_ptr->get_empty_related_viewgrams(view_seg_num, symmetries_ptr);

      {
	const double start_time = frame_defs.get_start_time(current_frame_num);
	const double end_time = frame_defs.get_end_time(current_frame_num);
        normatten_viewgrams.fill(1.F);
        normalisation_ptr->apply(normatten_viewgrams,start_time,end_time);
        
        if (!is_null_ptr(normatten_projdata_ptr))
	  normatten_projdata_ptr->set_related_viewgrams(normatten_viewgrams);
      }

      // ** now compute scatter **
      /*RelatedViewgrams<float>*/ scatter_viewgrams = normatten_viewgrams;
      if (do_scatter)
      {
        // scatter = trues_emission * norm * atten - fully_precorrected_emission

	if (!is_null_ptr(trues_projdata_ptr))
	  {
	    trues_viewgrams = trues_projdata_ptr->get_related_viewgrams(view_seg_num, symmetries_ptr);
	  }
	else
	  {
	    randoms_viewgrams = randoms_projdata_ptr->get_related_viewgrams(view_seg_num, symmetries_ptr);
	    already_read_randoms = true;
	    trues_viewgrams = prompts_projdata_ptr->get_related_viewgrams(view_seg_num, symmetries_ptr);
	    trues_viewgrams -= randoms_viewgrams;
	  }
        scatter_viewgrams *= 
	    trues_viewgrams;
        scatter_viewgrams -= 
          precorrected_projdata_ptr->get_related_viewgrams(view_seg_num, symmetries_ptr);
        
        scatter_projdata_ptr->set_related_viewgrams(scatter_viewgrams);
      }
      else
      {
	if (is_null_ptr(scatter_projdata_ptr))
	  scatter_viewgrams.fill(0);
	else
	  scatter_viewgrams = 
	    scatter_projdata_ptr->get_related_viewgrams(view_seg_num, symmetries_ptr);
      }

      if (do_Shifted_Poisson)
      {
        if (!already_read_randoms)
	  {
	    randoms_viewgrams =
	      randoms_projdata_ptr->get_related_viewgrams(view_seg_num, symmetries_ptr);
	  }

        // multiply with 2 for Shifted Poisson
        randoms_viewgrams *= 2;

        {
          // numerator of Shifted_Poisson is trues+ 2*randoms

          /*RelatedViewgrams<float>*/ Shifted_Poisson_numerator_viewgrams =
            Shifted_Poisson_numerator_projdata_ptr->get_empty_related_viewgrams(view_seg_num, symmetries_ptr);
          Shifted_Poisson_numerator_viewgrams += 
            trues_projdata_ptr->get_related_viewgrams(view_seg_num, symmetries_ptr);
          Shifted_Poisson_numerator_viewgrams += randoms_viewgrams;
          Shifted_Poisson_numerator_projdata_ptr->set_related_viewgrams(Shifted_Poisson_numerator_viewgrams);
        }
        {
          // denominator of Shifted_Poisson is scatter+ 2*randoms*norm*atten

          randoms_viewgrams *= normatten_viewgrams;
          /*RelatedViewgrams<float>*/ Shifted_Poisson_denominator_viewgrams = scatter_viewgrams;
          Shifted_Poisson_denominator_viewgrams += randoms_viewgrams;
          Shifted_Poisson_denominator_projdata_ptr->set_related_viewgrams(Shifted_Poisson_denominator_viewgrams);
        }
	// we force re-reading the randoms as we modified them above
        already_read_randoms = false;
      }
    
      if (do_prompts)
      {
        if (!already_read_randoms)
	  {
	    randoms_viewgrams =
	      randoms_projdata_ptr->get_related_viewgrams(view_seg_num, symmetries_ptr);
	  }
               
        {
          // denominator of prompts is scatter+ randoms*norm*atten

          randoms_viewgrams *= normatten_viewgrams;
          /*RelatedViewgrams<float>*/ prompts_denominator_viewgrams = scatter_viewgrams;
          prompts_denominator_viewgrams += randoms_viewgrams;
          prompts_denominator_projdata_ptr->set_related_viewgrams(prompts_denominator_viewgrams);
        }
      }

    }
        
  }
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
  PrepareProjData application( argc==2 ? argv[1] : 0);
 
  if (argc!=2)
    {
      cerr << "Corresponding .par file input \n"
	   << application.parameter_info() << endl;
    }
    

  CPUTimer timer;
  timer.start();

  application.doit();
 
  timer.stop();
  cerr << "CPU time : " << timer.value() << "secs" << endl;
  return EXIT_SUCCESS;

}
