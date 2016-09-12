#include "stir/scatter/SingleScatterSimulation.h"

START_NAMESPACE_STIR

const char * const
SingleScatterSimulation::registered_name =
        "Single Scatter Simulation";


SingleScatterSimulation::
SingleScatterSimulation() :
    base_type()
{
    this->set_defaults();
}

SingleScatterSimulation::
SingleScatterSimulation(const std::string& parameter_filename)
{
    this->initialise(parameter_filename);
}

SingleScatterSimulation::
~SingleScatterSimulation()
{}


void
SingleScatterSimulation::
initialise_keymap()
{
    base_type::initialise_keymap();
    //    this->parser.add_start_key("Single Scatter Simulation");
    //    this->parser.add_stop_key("end Single Scatter Simulation");
}

void
SingleScatterSimulation::
initialise(const std::string& parameter_filename)
{
    if (parameter_filename.size() == 0)
    {
        this->set_defaults();
        this->ask_parameters();
    }
    else
    {
        this->set_defaults();
        if (!this->parse(parameter_filename.c_str()))
        {
            error("Error parsing input file %s, exiting", parameter_filename.c_str());
        }
    }
}

void
SingleScatterSimulation::
set_defaults()
{
    base_type::set_defaults();
}

void
SingleScatterSimulation::
ask_parameters()
{

}

std::string
SingleScatterSimulation::
method_info() const
{

}


END_NAMESPACE_STIR

