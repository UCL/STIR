//
// $Id$
//

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
  output filename := precorrected.s
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
  ;attenuation image filename := attenuation_image.hv
  
  ; forward projector used to estimate attenuation factors, defaults to Ray Tracing
  ;forward_projector type := Ray Tracing

  ; scatter term to be subtracted AFTER norm+atten correction
  ; defaults to 0
  ;scatter projdata filename := scatter.hs
END:= 
\endverbatim

  \author Kris Thielemans

  $Date$
  $Revision$
*/


#include "utilities.h"
#include "interfile.h"
#include "CPUTimer.h"
#include "ProjDataFromStream.h"
#include "VoxelsOnCartesianGrid.h"
#include "RelatedViewgrams.h"
#include "tomo/ParsingObject.h"
#include "tomo/Succeeded.h"
#include "tomo/recon_buildblock/BinNormalisationFromProjData.h"
#include "tomo/recon_buildblock/TrivialBinNormalisation.h"

#include "ArrayFunction.h"
#ifndef USE_PMRT
#include "recon_buildblock/ForwardProjectorByBinUsingRayTracing.h"
#else
#include "recon_buildblock/ForwardProjectorByBinUsingProjMatrixByBin.h"
#include "recon_buildblock/ProjMatrixByBinUsingRayTracing.h"
#endif
//#include "display.h"

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





// note: apply_or_undo_correction==true means: apply it
static void
correct_projection_data(ProjData& output_projdata, const ProjData& input_projdata,
			const bool use_data_or_set_to_1,
			const bool apply_or_undo_correction,
                        const shared_ptr<ProjData>& scatter_projdata_ptr,
			shared_ptr<DiscretisedDensity<3,float> >& attenuation_image_ptr,
			const shared_ptr<ForwardProjectorByBin>& forward_projector_ptr,
			const BinNormalisation& normalisation,
                        const shared_ptr<ProjData>& randoms_projdata_ptr
                        )
{
  const bool do_attenuation = attenuation_image_ptr.use_count() != 0;
  const bool do_scatter = scatter_projdata_ptr.use_count() != 0;
  const bool do_randoms = randoms_projdata_ptr.use_count() != 0;

  /* unfortunately, even when do_attenuation==false, we need the 
     attenuation_image and the forward projector, just to get the 
     DataSymmetriesForViewSegmentNumbers (for making RelatedViewgrams objects)
     TODO, get rid of this
     */
  if (!do_attenuation)
    attenuation_image_ptr = 
      new VoxelsOnCartesianGrid<float>(*input_projdata.get_proj_data_info_ptr());

  shared_ptr<DataSymmetriesForViewSegmentNumbers> symmetries_ptr=
    forward_projector_ptr->get_symmetries_used()->clone();

  // TODO somehow find basic range for loop
  for (int segment_num = 0; segment_num <= output_projdata.get_max_segment_num() ; segment_num++)
  {
    cerr<<endl<<"Processing segment #"<<segment_num<<endl;
    for (int view_num=0; view_num<=input_projdata.get_num_views()/4; ++view_num)
    {    
      const ViewSegmentNumbers view_seg_nums(view_num,segment_num);
      // ** first fill in the data **
      
      RelatedViewgrams<float> 
        viewgrams = output_projdata.get_empty_related_viewgrams(ViewSegmentNumbers(view_num,segment_num),
	                          symmetries_ptr, false);
      if (use_data_or_set_to_1)
      {
	// Unfortunately, segment range in output_projdata and input_projdata can be
        // different. So, we cannot simply use 
        // viewgrams = input_projdata.get_related_viewgrams
        // as this would give it the wrong proj_data_info_ptr 
        // (resulting in problems when setting the viewgrams in output_projdata).
        // The trick relies on calling Array::operator+= instead of 
        // Viewgrams::operator=
        viewgrams += 
          input_projdata.get_related_viewgrams(view_seg_nums,
	                            symmetries_ptr, false);
      }	  
      else
      {
        viewgrams.fill(1.F);
      }
      
	      
      // display(viewgrams);      

      if (do_scatter && !apply_or_undo_correction)
      {
        viewgrams += 
          scatter_projdata_ptr->get_related_viewgrams(view_seg_nums,
	                                              symmetries_ptr, false);
      }

      if (do_randoms && apply_or_undo_correction)
      {
        viewgrams -= 
          randoms_projdata_ptr->get_related_viewgrams(view_seg_nums,
	                                              symmetries_ptr, false);
      }

      // ** normalisation **
      if (apply_or_undo_correction)
      {
	normalisation.apply(viewgrams);
      }
      else
      {
        normalisation.undo(viewgrams);
      }


      // ** attenuation ** 
      if (do_attenuation)
      {	
	RelatedViewgrams<float> attenuation_viewgrams = 
	  output_projdata.get_empty_related_viewgrams(view_seg_nums,
	                                  symmetries_ptr, false);	
	
	forward_projector_ptr->forward_project(attenuation_viewgrams, *attenuation_image_ptr);
	
	// TODO cannot use std::transform ?
	for (RelatedViewgrams<float>::iterator viewgrams_iter = 
	             attenuation_viewgrams.begin();
	     viewgrams_iter != attenuation_viewgrams.end();
	     ++viewgrams_iter)
	{
	  in_place_exp(*viewgrams_iter);
	}
	if (apply_or_undo_correction)
          viewgrams *= attenuation_viewgrams;
        else
          viewgrams /= attenuation_viewgrams;

      } // do_attenuation
      

      if (do_scatter && apply_or_undo_correction)
      {
        viewgrams -= 
          scatter_projdata_ptr->get_related_viewgrams(view_seg_nums,
	                                              symmetries_ptr, false);
      }

      if (do_randoms && !apply_or_undo_correction)
      {
        viewgrams += 
          randoms_projdata_ptr->get_related_viewgrams(view_seg_nums,
	                                              symmetries_ptr, false);
      }
      
      if (!(output_projdata.set_related_viewgrams(viewgrams) == Succeeded::yes))
        error("Error set_related_viewgrams\n");            
      
    }
        
  }
}    

// TODO most of this is identical to ReconstructionParameters, so make a common class
class CorrectProjDataParameters : public ParsingObject
{
public:

  CorrectProjDataParameters(const char * const par_filename);

  // shared_ptrs such that they clean up automatically at exit
  shared_ptr<ProjData> input_projdata_ptr;
  shared_ptr<ProjData> scatter_projdata_ptr;
  shared_ptr<ProjData> randoms_projdata_ptr;
  shared_ptr<ProjDataFromStream> output_projdata_ptr;
  shared_ptr<BinNormalisation> normalisation_ptr;
  shared_ptr<DiscretisedDensity<3,float> > attenuation_image_ptr;
  shared_ptr<ForwardProjectorByBin> forward_projector_ptr;
  bool apply_or_undo_correction;
  bool use_data_or_set_to_1;  
  int max_segment_num_to_process;

private:

  virtual void set_defaults();
  virtual void initialise_keymap();
  string input_filename;
  string output_filename;
  string scatter_projdata_filename;
  string atten_image_filename;
  string norm_filename;  
  string randoms_projdata_filename;
};

void 
CorrectProjDataParameters::
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

#ifndef USE_PMRT
  forward_projector_ptr =
    new ForwardProjectorByBinUsingRayTracing;
#else
  shared_ptr<ProjMatrixByBin> PM = 
    new  ProjMatrixByBinUsingRayTracing;
  forward_projector_ptr =
    new ForwardProjectorByBinUsingProjMatrixByBin(PM); 
#endif
}

void 
CorrectProjDataParameters::
initialise_keymap()
{
  parser.add_start_key("correct_projdata Parameters");
  parser.add_key("input file",&input_filename);
  parser.add_key("output filename",&output_filename);
  parser.add_key("maximum absolute segment number to process", &max_segment_num_to_process);
 
  parser.add_key("use data (1) or set to one (0)", &use_data_or_set_to_1);
  parser.add_key("apply (1) or undo (0) correction", &apply_or_undo_correction);
  parser.add_parsing_key("Bin Normalisation type", &normalisation_ptr);
  parser.add_key("randoms projdata filename", &randoms_projdata_filename);
  //parser.add_key("Normalisation filename", &norm_filename);
  parser.add_key("attenuation image filename", &atten_image_filename);
  parser.add_parsing_key("forward projector type", &forward_projector_ptr);
  parser.add_key("scatter_projdata_filename", &scatter_projdata_filename);
  parser.add_stop_key("END");
}

CorrectProjDataParameters::
CorrectProjDataParameters(const char * const par_filename)
{
  set_defaults();
  if (par_filename!=0)
    parse(par_filename) ;
  else
    ask_parameters();
  /*
  if (norm_filename!="" && norm_filename != "1")
    normalisation_ptr = new BinNormalisationFromProjData(norm_filename);
  else
    normalisation_ptr = new TrivialBinNormalisation;
  */
  input_projdata_ptr = ProjData::read_from_file(input_filename);

  if (scatter_projdata_filename!="" && scatter_projdata_filename != "0")
    scatter_projdata_ptr = ProjData::read_from_file(scatter_projdata_filename);

  if (randoms_projdata_filename!="" && randoms_projdata_filename != "0")
    randoms_projdata_ptr = ProjData::read_from_file(randoms_projdata_filename);

  const int max_segment_num_available =
    input_projdata_ptr->get_max_segment_num();
  if (max_segment_num_to_process<0 ||
      max_segment_num_to_process > max_segment_num_available)
    max_segment_num_to_process = max_segment_num_available;

  // construct output_projdata
  {
    ProjDataInfo*  new_data_info_ptr= 
      input_projdata_ptr->get_proj_data_info_ptr()->clone();
    
    new_data_info_ptr->reduce_segment_range(-max_segment_num_to_process, 
                                            max_segment_num_to_process);
    
    iostream * output_stream_ptr = 
      new fstream (output_filename.c_str(), ios::out| ios::binary);
    if (!output_stream_ptr->good())
    {
      error("error opening file %s\n",output_filename.c_str());
    }
    
    
    output_projdata_ptr = new ProjDataFromStream(new_data_info_ptr,output_stream_ptr);
    
    write_basic_interfile_PDFS_header(output_filename, *output_projdata_ptr);
  }

  // read attenuation data
  if(atten_image_filename!="0" && atten_image_filename!="")
  {
    attenuation_image_ptr = 
      DiscretisedDensity<3,float>::read_from_file(atten_image_filename.c_str());
        
    cerr << "WARNING: attenuation image data are supposed to be in units cm^-1\n"
      "Reference: water has mu .096 cm^-1" << endl;
    cerr<< "Max in attenuation image:" 
      << attenuation_image_ptr->find_max() << endl;
#ifndef NORESCALE
    /*
      cerr << "WARNING: multiplying attenuation image by x-voxel size "
      << " to correct for scale factor in forward projectors...\n";
    */
    // projectors work in pixel units, so convert attenuation data 
    // from cm^-1 to pixel_units^-1
    const float rescale = 
      dynamic_cast<VoxelsOnCartesianGrid<float> *>(attenuation_image_ptr.get())->
      get_voxel_size().x()/10;
#else
    const float rescale = 
      10.F;
#endif
    *attenuation_image_ptr *= rescale;

    forward_projector_ptr->set_up(output_projdata_ptr->get_proj_data_info_ptr()->clone(),
				  attenuation_image_ptr);
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
  CorrectProjDataParameters parameters( argc==2 ? argv[1] : 0);
 
  if (argc!=2)
    {
      cerr << "Corresponding .par file input \n"
	   << parameters.parameter_info() << endl;
    }
    

  CPUTimer timer;
  timer.start();

  correct_projection_data(*parameters.output_projdata_ptr, *parameters.input_projdata_ptr, 
			  parameters.use_data_or_set_to_1, parameters.apply_or_undo_correction,
                          parameters.scatter_projdata_ptr,
			  parameters.attenuation_image_ptr,  
			  parameters.forward_projector_ptr,  
			  *parameters.normalisation_ptr,
                          parameters.randoms_projdata_ptr);
 
  timer.stop();
  cerr << "CPU time : " << timer.value() << "secs" << endl;
  return EXIT_SUCCESS;

}
