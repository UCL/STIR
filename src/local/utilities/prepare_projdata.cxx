//
// $Id$
//

/*!
  \file
  \ingroup utilities

  \brief A utility applying/undoing some corrections on projection data

  \author Kris Thielemans

  $Date$
  $Revision$
*/


#include "utilities.h"
#include "interfile.h"
#include "CPUTimer.h"
#include "ProjDataFromStream.h"
#include "RelatedViewgrams.h"
#include "TrivialDataSymmetriesForViewSegmentNumbers.h"
#include "tomo/ParsingObject.h"
#include "tomo/Succeeded.h"

#include "tomo/recon_buildblock/BinNormalisationFromProjData.h"
#include "tomo/recon_buildblock/TrivialBinNormalisation.h"

#include <string>
#include <iostream> 
#include <fstream>
#include <algorithm>

#ifndef TOMO_NO_NAMESPACES
using std::cerr;
using std::endl;
using std::fstream;
using std::ifstream;
using std::cout;
using std::string;
#endif

START_NAMESPACE_TOMO






ProjDataFromStream *
make_new_ProjDataFromStream(const string& output_filename, shared_ptr<ProjDataInfo>& proj_data_info_sptr)
{
  
  iostream * output_stream_ptr = 
    new fstream (output_filename.c_str(), ios::out| ios::binary);
  if (!output_stream_ptr->good())
  {
    error("error opening file %s\n",output_filename.c_str());
  }  
  
  ProjDataFromStream*  output_projdata_ptr = 
    new ProjDataFromStream(proj_data_info_sptr,output_stream_ptr);
  
  write_basic_interfile_PDFS_header(output_filename.c_str(), *output_projdata_ptr);

  return output_projdata_ptr;
}

class PrepareProjData : public ParsingObject
{
public:

  PrepareProjData(const char * const par_filename);
  void doit();

private:
  // shared_ptr's such that they clean up automatically at exit
  shared_ptr<ProjData> trues_projdata_ptr;
  shared_ptr<ProjData> precorrected_projdata_ptr;
  shared_ptr<ProjData> randoms_projdata_ptr;
  shared_ptr<BinNormalisation> normalisation_ptr;
  shared_ptr<ProjData> attenuation_projdata_ptr;
  
  
  shared_ptr<ProjData> normatten_projdata_ptr;
  shared_ptr<ProjData> scatter_projdata_ptr;
  shared_ptr<ProjData> Shifted_Poisson_numerator_projdata_ptr;
  shared_ptr<ProjData> Shifted_Poisson_denominator_projdata_ptr;
  shared_ptr<ProjData> prompts_denominator_projdata_ptr;
  
  int max_segment_num_to_process;
  bool do_Shifted_Poisson;
  bool do_prompts;
private:

  virtual void set_defaults();
  virtual void initialise_keymap();

  string attenuation_projdata_filename;
  string norm_filename;
  string trues_projdata_filename;
  string precorrected_projdata_filename;
  string randoms_projdata_filename;
  
  string normatten_projdata_filename;
  string scatter_projdata_filename;
  string Shifted_Poisson_numerator_projdata_filename;
  string Shifted_Poisson_denominator_projdata_filename;
  string prompts_denominator_projdata_filename;
  
};

void 
PrepareProjData::
set_defaults()
{
  trues_projdata_ptr = 0;
  precorrected_projdata_ptr = 0;
  randoms_projdata_ptr = 0;
  normalisation_ptr = 0;
  attenuation_projdata_ptr = 0;
  max_segment_num_to_process = -1;
  attenuation_projdata_filename = "";
  norm_filename = "";
  trues_projdata_filename = "";
  precorrected_projdata_filename = "";
  randoms_projdata_filename = "";
  
  normatten_projdata_filename = "";
  scatter_projdata_filename = "";
  Shifted_Poisson_numerator_projdata_filename = "";
  Shifted_Poisson_denominator_projdata_filename = "";
  
}

void 
PrepareProjData::
initialise_keymap()
{
  parser.add_start_key("Prepare projdata Parameters");
  parser.add_key("attenuation_projdata_filename", &attenuation_projdata_filename);
  parser.add_key("Normalisation projdata filename", &norm_filename);
  parser.add_key("trues_projdata_filename", &trues_projdata_filename);
  parser.add_key("precorrected_projdata_filename", &precorrected_projdata_filename);
  parser.add_key("randoms_projdata_filename", &randoms_projdata_filename);
  
  
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

  if (norm_filename!="")
    normalisation_ptr = new BinNormalisationFromProjData(norm_filename);
  else
    normalisation_ptr = new TrivialBinNormalisation;

  trues_projdata_ptr = ProjData::read_from_file(trues_projdata_filename.c_str());
  precorrected_projdata_ptr = ProjData::read_from_file(precorrected_projdata_filename.c_str());
  attenuation_projdata_ptr = ProjData::read_from_file(attenuation_projdata_filename.c_str());
  do_Shifted_Poisson = 
    randoms_projdata_filename.size()!=0 &&
    Shifted_Poisson_numerator_projdata_filename.size() != 0;
  do_prompts =
    randoms_projdata_filename.size()!=0 &&
    prompts_denominator_projdata_filename.size() != 0;

  if (do_Shifted_Poisson || do_prompts)
     randoms_projdata_ptr = ProjData::read_from_file(randoms_projdata_filename.c_str());
  const int max_segment_num_available =
    trues_projdata_ptr->get_max_segment_num();
  if (max_segment_num_to_process<0 ||
      max_segment_num_to_process > max_segment_num_available)
    max_segment_num_to_process = max_segment_num_available;

  // construct output projdata
  {
    shared_ptr<ProjDataInfo>  new_data_info_ptr= 
      trues_projdata_ptr->get_proj_data_info_ptr()->clone();
    new_data_info_ptr->reduce_segment_range(-max_segment_num_to_process, 
                                            max_segment_num_to_process);
    

    normatten_projdata_ptr = make_new_ProjDataFromStream(normatten_projdata_filename, new_data_info_ptr);
    scatter_projdata_ptr = make_new_ProjDataFromStream(scatter_projdata_filename, new_data_info_ptr);
    if (do_Shifted_Poisson)
    {
      Shifted_Poisson_numerator_projdata_ptr = make_new_ProjDataFromStream(Shifted_Poisson_numerator_projdata_filename, new_data_info_ptr);
      Shifted_Poisson_denominator_projdata_ptr = make_new_ProjDataFromStream(Shifted_Poisson_denominator_projdata_filename, new_data_info_ptr);
    }
    if (do_prompts)
    {
      prompts_denominator_projdata_ptr = make_new_ProjDataFromStream(prompts_denominator_projdata_filename, new_data_info_ptr);
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
  RelatedViewgrams<float> randoms_viewgrams;
  RelatedViewgrams<float> Shifted_Poisson_numerator_viewgrams;
  RelatedViewgrams<float> Shifted_Poisson_denominator_viewgrams;
  RelatedViewgrams<float> prompts_denominator_viewgrams;

  // TODO somehow find basic range for loop, in this case there are no symmetries used
  for (int segment_num = -max_segment_num_to_process; segment_num <= max_segment_num_to_process; segment_num++)
  {
    cerr<<endl<<"Processing segment #"<<segment_num<<endl;
    for (int view_num=normatten_projdata_ptr->get_min_view_num(); view_num<=normatten_projdata_ptr->get_max_view_num(); ++view_num)
    {    
      const ViewSegmentNumbers view_seg_num(view_num,segment_num);

      // ** first fill in  normatten **
      
      /*RelatedViewgrams<float>*/ normatten_viewgrams = 
        normatten_projdata_ptr->get_empty_related_viewgrams(view_seg_num, symmetries_ptr, false);

      {
        normatten_viewgrams.fill(1.F);
        normalisation_ptr->apply(normatten_viewgrams);
        
        normatten_viewgrams *= 
          attenuation_projdata_ptr->get_related_viewgrams(view_seg_num, symmetries_ptr, false);
        
        normatten_projdata_ptr->set_related_viewgrams(normatten_viewgrams);
      }

      // ** now compute scatter **
      
      /*RelatedViewgrams<float>*/ scatter_viewgrams = normatten_viewgrams;
      {
        // scatter = trues_emission * norm * atten - fully_precorrected_emission

        scatter_viewgrams *= 
          trues_projdata_ptr->get_related_viewgrams(view_seg_num, symmetries_ptr, false);
        scatter_viewgrams -= 
          precorrected_projdata_ptr->get_related_viewgrams(view_seg_num, symmetries_ptr, false);
        
        scatter_projdata_ptr->set_related_viewgrams(scatter_viewgrams);
      }

      if (do_Shifted_Poisson)
      {
        /*RelatedViewgrams<float>*/ randoms_viewgrams =
          randoms_projdata_ptr->get_related_viewgrams(view_seg_num, symmetries_ptr, false);
        // multiply with 2 for Shifted Poisson
        randoms_viewgrams *= 2;

        {
          // numerator of Shifted_Poisson is trues+ 2*randoms

          /*RelatedViewgrams<float>*/ Shifted_Poisson_numerator_viewgrams =
            Shifted_Poisson_numerator_projdata_ptr->get_empty_related_viewgrams(view_seg_num, symmetries_ptr, false);
          Shifted_Poisson_numerator_viewgrams += 
            trues_projdata_ptr->get_related_viewgrams(view_seg_num, symmetries_ptr, false);
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
      }
    
      if (do_prompts)
      {
        /*RelatedViewgrams<float>*/ randoms_viewgrams =
          randoms_projdata_ptr->get_related_viewgrams(view_seg_num, symmetries_ptr, false);
               
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



END_NAMESPACE_TOMO

USING_NAMESPACE_TOMO

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
