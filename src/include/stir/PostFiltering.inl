
START_NAMESPACE_STIR

template <class DataT>
PostFiltering<DataT>::PostFiltering()
{
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
    filter_sptr.reset();
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
Succeeded
PostFiltering<DataT>::process_data(DataT& arg)
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
