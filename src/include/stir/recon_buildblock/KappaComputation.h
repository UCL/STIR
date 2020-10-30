//
// Created by Robert Twyman on 30/10/2020.
//

#ifndef STIR_KAPPACOMPUTATION_H
#define STIR_KAPPACOMPUTATION_H

#include "stir/info.h"
#include "stir/IO/OutputFileFormat.h"
#include "stir/IO/read_from_file.h"
#include "stir/is_null_ptr.h"
#include "stir/recon_buildblock/GeneralisedObjectiveFunction.h"
#include "stir/recon_buildblock/GeneralisedPrior.h"
#include "stir/recon_buildblock/PoissonLogLikelihoodWithLinearModelForMeanAndProjData.h"



START_NAMESPACE_STIR


class Succeeded;

template <typename TargetT>
class KappaComputation:
        public ParsingObject
{
    //All methods need documenting
public:
    KappaComputation();
    void set_defaults();
    void process_data();

protected:
    shared_ptr<TargetT> current_image_estimate_sptr;
    shared_ptr<TargetT> template_image_sptr;

    shared_ptr<GeneralisedObjectiveFunction<TargetT> >  objective_function_sptr;
    shared_ptr<OutputFileFormat<TargetT> > output_file_format_sptr;

    void compute_kappa_at_current_image_estimate();
    void compute_kappa_with_approximate();

private:
    std::string current_image_estimate_filename;
    std::string template_image_filename;
    std::string kappa_filename;
    void initialise_keymap();
    bool post_processing();

    void sqrt_image(TargetT& output_image_sptr);
};

END_NAMESPACE_STIR
#endif //STIR_KAPPACOMPUTATION_H
