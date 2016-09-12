/*!
  \file
  \ingroup scatter
  \brief Definition of class stir::SingleScatterSimulation

  \author Nikos Efthimiou
  \author Kris Thielemans
*/

#ifndef __stir_scatter_SingleScatterSimulation_H__
#define __stir_scatter_SingleScatterSimulation_H__

#include "stir/Succeeded.h"
#include "stir/scatter/ScatterSimulation.h"
#include "stir/shared_ptr.h"
#include "stir/RegisteredParsingObject.h"


START_NAMESPACE_STIR

class SingleScatterSimulation : public
        RegisteredParsingObject<
        SingleScatterSimulation,
        ScatterSimulation,
        ScatterSimulation >
{
private:
    typedef RegisteredParsingObject<
    SingleScatterSimulation,
    ScatterSimulation,
    ScatterSimulation > base_type;
public:

    //! Name which will be used when parsing a OSMAPOSLReconstruction object
    static const char * const registered_name;

    //!
    //! \brief ScatterSimulation
    //! \details Default constructor
    SingleScatterSimulation();

    //!
    //! \brief ScatterSimulation
    //! \param parameter_filename
    //! \details Constructor with initialisation from parameter file
    explicit
    SingleScatterSimulation(const std::string& parameter_filename);

    virtual ~SingleScatterSimulation();

    //    virtual Succeeded
    //    process_data();

    //! gives method information
    virtual std::string method_info() const;

    //! prompts the user to enter parameter values manually
    virtual void ask_parameters();

protected:

    void initialise(const std::string& parameter_filename);

    virtual void set_defaults();
    virtual void initialise_keymap();

    //! used to check acceptable parameter ranges, etc...
//    virtual bool post_processing();


    //!
    //! \brief simulate_for_one_scatter_point
    //! \param scatter_point_num
    //! \param det_num_A
    //! \param det_num_B
    //! \return
    //! \details This funtion used to be ScatterEstimationByBin::
    //! single_scatter_estimate_for_one_scatter_point()
    float
    simulate_for_one_scatter_point(const std::size_t scatter_point_num,
                                                  const unsigned det_num_A,
                                                  const unsigned det_num_B);

    virtual void
    actual_scatter_estimate(double& scatter_ratio_singles,
                            const unsigned det_num_A,
                            const unsigned det_num_B);



};

END_NAMESPACE_STIR

#endif
