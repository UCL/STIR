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



// A do-nothing class for normalisation
class Normalisation
{
public:
  virtual ~Normalisation() {}
  virtual void apply(RelatedViewgrams<float>&) const= 0;
  virtual void undo(RelatedViewgrams<float>&) const = 0; 
};

class NoNormalisation : public Normalisation
{
public:
  virtual void apply(RelatedViewgrams<float>&) const {}
  virtual void undo(RelatedViewgrams<float>&) const {}
};

class ProfileNormalisation : public Normalisation
{
public:
  ProfileNormalisation(const string& filename)
    : profile(  Array<1,float>(-40,39) )
  {
    ifstream profile_data(filename.c_str());
    for (int i=profile.get_min_index(); i<=profile.get_max_index(); ++i)
      profile_data >> profile[i];
    if (!profile_data)
      error("Error reading profile %s\n", filename.c_str());
  }

  virtual void apply(RelatedViewgrams<float>& viewgrams) const 
  {
    RelatedViewgrams<float>::iterator viewgrams_iter = 
      viewgrams.begin();
    for (; 
	 viewgrams_iter != viewgrams.end();
	 ++viewgrams_iter)
      {
	for (int ax_pos_num=viewgrams_iter->get_min_index();
	     ax_pos_num<=viewgrams_iter->get_max_index();
	     ++ax_pos_num)
	  {
	    (*viewgrams_iter)[ax_pos_num] *= profile;
	  }
      }
  }

  virtual void undo(RelatedViewgrams<float>& viewgrams) const 
    {
    RelatedViewgrams<float>::iterator viewgrams_iter = 
      viewgrams.begin();
    for (; 
	 viewgrams_iter != viewgrams.end();
	 ++viewgrams_iter)
      {
	for (int ax_pos_num=viewgrams_iter->get_min_index();
	     ax_pos_num<=viewgrams_iter->get_max_index();
	     ++ax_pos_num)
	  {
	    (*viewgrams_iter)[ax_pos_num] /= profile;
	  }
      }
    }
  private:
  Array<1,float> profile;
};

class NormalisationFromProjData : public Normalisation
{
public:
  NormalisationFromProjData(const string& filename)
    : norm_proj_data_ptr(ProjData::read_from_file(filename.c_str()))
  {}
  NormalisationFromProjData(const shared_ptr<ProjData>& norm_proj_data_ptr)
    : norm_proj_data_ptr(norm_proj_data_ptr)
  {}

  virtual void apply(RelatedViewgrams<float>& viewgrams) const 
  {
    const ViewSegmentNumbers vs_num=viewgrams.get_basic_view_segment_num();
    const DataSymmetriesForViewSegmentNumbers * symmetries_ptr =
      viewgrams.get_symmetries_ptr();
    viewgrams *= 
      norm_proj_data_ptr->get_related_viewgrams(vs_num,symmetries_ptr->clone(), false);
  }

  virtual void undo(RelatedViewgrams<float>& viewgrams) const 
  {
    const ViewSegmentNumbers vs_num=viewgrams.get_basic_view_segment_num();
    const DataSymmetriesForViewSegmentNumbers * symmetries_ptr =
      viewgrams.get_symmetries_ptr();
    viewgrams /= 
      norm_proj_data_ptr->get_related_viewgrams(vs_num,symmetries_ptr->clone(), false);

  }
private:
  shared_ptr<ProjData> norm_proj_data_ptr;
};



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
  shared_ptr<ProjData> raw_projdata_ptr;
  shared_ptr<ProjData> precorrected_projdata_ptr;
  shared_ptr<ProjData> randoms_projdata_ptr;
  shared_ptr<Normalisation> normalisation_ptr;
  shared_ptr<ProjData> attenuation_projdata_ptr;
  
  
  shared_ptr<ProjData> normatten_projdata_ptr;
  shared_ptr<ProjData> scatter_projdata_ptr;
  shared_ptr<ProjData> Shifted_Poisson_numerator_projdata_ptr;
  shared_ptr<ProjData> Shifted_Poisson_denominator_projdata_ptr;
  
  int max_segment_num_to_process;
  bool do_Shifted_Poisson;
private:

  virtual void set_defaults();
  virtual void initialise_keymap();

  string attenuation_projdata_filename;
  string norm_filename;
  string raw_projdata_filename;
  string precorrected_projdata_filename;
  string randoms_projdata_filename;
  
  string normatten_projdata_filename;
  string scatter_projdata_filename;
  string Shifted_Poisson_numerator_projdata_filename;
  string Shifted_Poisson_denominator_projdata_filename;
  
};

void 
PrepareProjData::
set_defaults()
{
  raw_projdata_ptr = 0;
  precorrected_projdata_ptr = 0;
  randoms_projdata_ptr = 0;
  normalisation_ptr = 0;
  attenuation_projdata_ptr = 0;
  max_segment_num_to_process = -1;
  attenuation_projdata_filename = "";
  norm_filename = "";
  raw_projdata_filename = "";
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
  parser.add_key("raw_projdata_filename", &raw_projdata_filename);
  parser.add_key("precorrected_projdata_filename", &precorrected_projdata_filename);
  parser.add_key("randoms_projdata_filename", &randoms_projdata_filename);
  
  
  parser.add_key("normatten_projdata_filename", &normatten_projdata_filename);
  parser.add_key("scatter_projdata_filename", &scatter_projdata_filename);
  parser.add_key("Shifted_Poisson_numerator_projdata_filename", &Shifted_Poisson_numerator_projdata_filename);
  parser.add_key("Shifted_Poisson_denominator_projdata_filename", &Shifted_Poisson_denominator_projdata_filename);
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
    normalisation_ptr = new NormalisationFromProjData(norm_filename);
  else
    normalisation_ptr = new NoNormalisation;

  raw_projdata_ptr = ProjData::read_from_file(raw_projdata_filename.c_str());
  precorrected_projdata_ptr = ProjData::read_from_file(precorrected_projdata_filename.c_str());
  attenuation_projdata_ptr = ProjData::read_from_file(attenuation_projdata_filename.c_str());
  do_Shifted_Poisson  = randoms_projdata_filename.size()!=0;
  if (do_Shifted_Poisson)
     randoms_projdata_ptr = ProjData::read_from_file(randoms_projdata_filename.c_str());
  const int max_segment_num_available =
    raw_projdata_ptr->get_max_segment_num();
  if (max_segment_num_to_process<0 ||
      max_segment_num_to_process > max_segment_num_available)
    max_segment_num_to_process = max_segment_num_available;

  // construct output projdata
  {
    shared_ptr<ProjDataInfo>  new_data_info_ptr= 
      raw_projdata_ptr->get_proj_data_info_ptr()->clone();
    new_data_info_ptr->reduce_segment_range(-max_segment_num_to_process, 
                                            max_segment_num_to_process);
    

    normatten_projdata_ptr = make_new_ProjDataFromStream(normatten_projdata_filename, new_data_info_ptr);
    scatter_projdata_ptr = make_new_ProjDataFromStream(scatter_projdata_filename, new_data_info_ptr);
    if (do_Shifted_Poisson)
    {
      Shifted_Poisson_numerator_projdata_ptr = make_new_ProjDataFromStream(Shifted_Poisson_numerator_projdata_filename, new_data_info_ptr);
      Shifted_Poisson_denominator_projdata_ptr = make_new_ProjDataFromStream(Shifted_Poisson_denominator_projdata_filename, new_data_info_ptr);
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
        // scatter = raw_emission * norm * atten - fully_precorrected_emission

        scatter_viewgrams *= 
          raw_projdata_ptr->get_related_viewgrams(view_seg_num, symmetries_ptr, false);
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
          // numerator of Shifted_Poisson is raw+ 2*randoms

          /*RelatedViewgrams<float>*/ Shifted_Poisson_numerator_viewgrams =
            Shifted_Poisson_numerator_projdata_ptr->get_empty_related_viewgrams(view_seg_num, symmetries_ptr, false);
          Shifted_Poisson_numerator_viewgrams += 
            raw_projdata_ptr->get_related_viewgrams(view_seg_num, symmetries_ptr, false);
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
