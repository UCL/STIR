/*
    Copyright (C) 2016 University College London
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
  \ingroup scatter
  \brief Definition of class stir::SingleScatterSimulation

  \author Nikos Efthimiou
*/

#ifndef __stir_scatter_SingleScatterSimulation_H__
#define __stir_scatter_SingleScatterSimulation_H__

#include "stir/Succeeded.h"
#include "stir/scatter/ScatterSimulation.h"
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

    //! Name which will be used when parsing a ScatterSimulation object
    static const char * const registered_name;

    //! Default constructor
    SingleScatterSimulation();

    //! Constructor with initialisation from parameter file
    explicit
    SingleScatterSimulation(const std::string& parameter_filename);

    virtual ~SingleScatterSimulation();

    virtual Succeeded process_data();

    //! gives method information
    virtual std::string method_info() const;

    //! prompts the user to enter parameter values manually
    virtual void ask_parameters();



    void initialise(const std::string& parameter_filename);

    virtual void set_defaults();
    virtual void initialise_keymap();

    virtual Succeeded set_up();

    //! used to check acceptable parameter ranges, etc...
    virtual bool post_processing();


protected:

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
