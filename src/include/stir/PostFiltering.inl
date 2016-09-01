
START_NAMESPACE_STIR

template <class DataT>
PostFiltering<DataT>::PostFiltering()
{
    set_defaults();
}

template <class DataT>
PostFiltering<DataT>::PostFiltering(const char * const par_filename)
{
    this->set_defaults();
    if (par_filename!=0)
    {
        if (parse(par_filename)==false)
            error("Exiting\n");
    }
    else
        ask_parameters();
}

template <class DataT>
void
PostFiltering<DataT>::set_defaults()
{
    filter_sptr.reset();
}

template <class DataT>
void
PostFiltering<DataT>::initialise_keymap()
{
    parser.add_start_key("PostFilteringParameters");
    parser.add_start_key("PostFiltering parameters");
    parser.add_parsing_key("PostFilter type", &filter_sptr);
    parser.add_stop_key("END PostFiltering Parameters");
}

template <class DataT>
bool
PostFiltering<DataT>::post_processing()
{
    return false;
}

template <class DataT>
Succeeded PostFiltering<DataT>::process_data(DataT& arg)
{
    return filter_sptr->apply(arg);
}

template <class DataT>
bool
PostFiltering<DataT>::is_filter_null()
{
    return is_null_ptr(filter_sptr);
}

END_NAMESPACE_STIR
