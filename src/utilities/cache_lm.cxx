//
//
/*!
  \file
  \ingroup listmode_utilities

\brief Program to bin listmode data to 3d sinograms

\see cache_lm for info on parameter file format

\author Nikos Efthimiou

 */

#include "stir/ProjDataInfoCylindricalNoArcCorr.h"
#include "stir/ProjData.h"
#include "stir/utilities.h"
#include "stir/ParsingObject.h"
#include "stir/listmode/ListModeData.h"
#include "stir/listmode/ListRecord.h"
#include "stir/listmode/CListEventCylindricalScannerWithDiscreteDetectors.h"
#include "stir/IO/read_from_file.h"
#include "stir/error.h"
#include "stir/FilePath.h"

using std::cerr;
using std::min;

USING_NAMESPACE_STIR

    class LmCache : public ParsingObject
{
public:
  LmCache(const char* const par_filename);

  int max_segment_num_to_process;

  shared_ptr<ListModeData> lm_data_sptr;

  void cache_listmode_file();

  std::string get_cache_filename(unsigned int file_id) const;

  std::string get_cache_path() const;

  Succeeded write_listmode_cache_file(unsigned int file_id) const;

private:
  void set_defaults() override;
  void initialise_keymap() override;
  bool post_processing() override;

  unsigned long int cache_size;
  std::string cache_path;
  std::string input_filename;
  std::string output_filename_prefix;
  bool store_prompts;
  int num_cache_files;
long int num_events_to_use;
  shared_ptr<ProjDataInfo> proj_data_info_sptr;
  std::vector<BinAndCorr> record_cache;
};


void LmCache::set_defaults()
{
  max_segment_num_to_process = -1;
  store_prompts = true;
  num_cache_files = 0;
  cache_size  = 150000000;
 num_events_to_use = -1;
}

void
LmCache::initialise_keymap()
{
  parser.add_start_key("lm_cache Parameters");
  parser.add_key("input file", &input_filename);
  parser.add_key("num_events_to_use", &num_events_to_use);
  // parser.add_key("frame_definition file", &frame_definition_filename);
  parser.add_key("output filename prefix", &output_filename_prefix);
  parser.add_key("maximum absolute segment number to process", &max_segment_num_to_process);
  // TODO can't do this yet
  // if (CListEvent::has_delayeds())
  {
    parser.add_key("Store 'prompts'", &store_prompts);
  }
  parser.add_stop_key("END");
}

bool
LmCache::post_processing()
{
  lm_data_sptr = read_from_file<ListModeData>(input_filename);

  const int num_rings = lm_data_sptr->get_scanner().get_num_rings();
  if (max_segment_num_to_process == -1)
    max_segment_num_to_process = num_rings - 1;
  else
    max_segment_num_to_process = min(max_segment_num_to_process, num_rings - 1);

  proj_data_info_sptr = lm_data_sptr->get_proj_data_info_sptr()->create_shared_clone()->create_single_tof_clone();
  proj_data_info_sptr->reduce_segment_range(-max_segment_num_to_process, max_segment_num_to_process);

  return false;
}

std::string
LmCache::get_cache_path() const
{
  if (this->cache_path.size() > 0)
    return this->cache_path;
  else
    return FilePath::get_current_working_directory();
}


LmCache::LmCache(const char* const par_filename)
{
  set_defaults();
  if (par_filename != 0)
    parse(par_filename);
  else
    ask_parameters();
}

std::string
LmCache::get_cache_filename(unsigned int file_id) const
{
  std::string cache_filename = "my_CACHE" + std::to_string(file_id) + ".bin";
  FilePath icache(cache_filename, false);
  icache.prepend_directory_name(this->get_cache_path());
  return icache.get_as_string();
}

Succeeded
LmCache::write_listmode_cache_file(
    unsigned int file_id) const
{
  const auto cache_filename = this->get_cache_filename(file_id);
  const bool with_add = false;

  {
    info("Storing Listmode cache to file \"" + cache_filename + "\".");
    // open the file, overwriting whatever was there before
    std::ofstream fout(cache_filename, std::ios::out | std::ios::binary | std::ios::trunc);
    if (!fout)
      error("Error opening cache file \"" + cache_filename + "\" for writing.");

    for (unsigned long int ie = 0; ie < record_cache.size(); ++ie)
      {
        Bin tmp = record_cache[ie].my_bin;
        if (with_add)
          tmp.set_bin_value(record_cache[ie].my_corr);
        fout.write((char*)&tmp, sizeof(Bin));
      }
    if (!fout)
      error("Error writing to cache file \"" + cache_filename + "\".");

    fout.close();
  }

  return Succeeded::yes;
}

void
LmCache::cache_listmode_file()
{

         // warning("Looking for existing cache files such as \"" + this->get_cache_filename(0) + "\".\n"
         //         + "We will be ignoring any time frame definitions as well as num_events_to_use!");
         // // find how many cache files there are
         // num_cache_files = 0;
         // while (true)
         //   {
         //     if (!FilePath::exists(this->get_cache_filename(this->num_cache_files)))
         //       break;
         //     ++this->num_cache_files;
         //   }
         // if (!this->num_cache_files)
         //   error("No cache files found.");
         // return ; // Stop here!!!

  num_cache_files = 0;

  info("Listmode reconstruction: Creating cache...", 2);

  record_cache.clear();
  try
    {
      record_cache.reserve(this->cache_size);
    }
  catch (...)
    {
      error("Listmode: cannot allocate cache for " + std::to_string(this->cache_size) + " records. Reduce cache size.");
    }

  this->lm_data_sptr->reset();
  const shared_ptr<ListRecord> record_sptr = this->lm_data_sptr->get_empty_record_sptr();

  double current_time = 0.;
  unsigned long int cached_events = 0;

  bool stop_caching = false;
  record_cache.reserve(this->cache_size);

  while (true) // keep caching across multiple files.
    {
      record_cache.clear();

      while (true) // Start for the current cache
        {
          if (this->lm_data_sptr->get_next_record(*record_sptr) == Succeeded::no)
            {
              stop_caching = true;
              break;
            }


          if (record_sptr->is_event() && record_sptr->event().is_prompt())
            {
              BinAndCorr tmp;
              tmp.my_bin.set_bin_value(1.0);
              record_sptr->event().get_bin(tmp.my_bin, *this->proj_data_info_sptr);

              if (tmp.my_bin.get_bin_value() != 1.0f
                  || tmp.my_bin.segment_num() < this->proj_data_info_sptr->get_min_segment_num()
                  || tmp.my_bin.segment_num() > this->proj_data_info_sptr->get_max_segment_num()
                  || tmp.my_bin.tangential_pos_num() < this->proj_data_info_sptr->get_min_tangential_pos_num()
                  || tmp.my_bin.tangential_pos_num() > this->proj_data_info_sptr->get_max_tangential_pos_num()
                  || tmp.my_bin.axial_pos_num() < this->proj_data_info_sptr->get_min_axial_pos_num(tmp.my_bin.segment_num())
                  || tmp.my_bin.axial_pos_num() > this->proj_data_info_sptr->get_max_axial_pos_num(tmp.my_bin.segment_num())
                  || tmp.my_bin.timing_pos_num() < this->proj_data_info_sptr->get_min_tof_pos_num()
                  || tmp.my_bin.timing_pos_num() > this->proj_data_info_sptr->get_max_tof_pos_num())
                {
                  continue;
                }
              try
                {
                  record_cache.push_back(tmp);
                  ++cached_events;
                }
              catch (...)
                {
                  // should never get here due to `reserve` statement above, but best to check...
                  error("Listmode: running out of memory for cache. Current size: "
                        + std::to_string(this->record_cache.size()) + " records");
                }

              if (record_cache.size() > 1 && record_cache.size() % 500000L == 0)
                info(boost::format("Cached Prompt Events (this cache): %1% ") % record_cache.size());

              if (this->num_events_to_use > 0)
                if (cached_events >= static_cast<std::size_t>(this->num_events_to_use))
                  {
                    stop_caching = true;
                    break;
                  }

              if (record_cache.size() == this->cache_size)
                break; // cache is full. go to next cache.
            }
        }

      if (write_listmode_cache_file(this->num_cache_files) == Succeeded::no)
        {
          error("Error writing cache file!");
        }
      ++this->num_cache_files;

      if (stop_caching)
        break;
    }
  info(boost::format("Cached Events: %1% ") % cached_events);
  return;

}


int
main(int argc, char* argv[])
{

  if (argc != 1 && argc != 2)
    {
      cerr << "Usage: " << argv[0] << " [par_file]\n";
      exit(EXIT_FAILURE);
    }
  LmCache lm_cache(argc == 2 ? argv[1] : 0);
  lm_cache.cache_listmode_file();

  return EXIT_SUCCESS;
}
