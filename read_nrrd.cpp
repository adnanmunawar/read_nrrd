//g++ read_nrrd.cpp -O3 -std=c++17 -o read_nrrd `pkg-config opencv --cflags --libs` -lboost_iostreams

#include <iostream>      // std :: cout
#include <unordered_map> // std :: unordered_map
#include <sstream>       // std :: stringstream
#include <fstream>       // std :: ifstream
#include <vector>        // std :: vector
#include <functional>    // std :: function
#include <regex>         // std :: regex
#include <algorithm>     // std :: copy_n
#include <numeric>       // std :: accumulate
#include <iterator>      // std :: ostream_iterator
#include <memory>        // std :: unique_ptr
#include <string>        // std :: stoi, std :: stof
#include <array>         // std :: array

#include <boost/iostreams/filtering_stream.hpp>
#include <boost/iostreams/filter/gzip.hpp>
#include <boost/iostreams/filter/bzip2.hpp>
#include <boost/iostreams/copy.hpp>

#include <opencv2/opencv.hpp>
// #include <opencv4/opencv2/opencv.hpp>


std :: regex value_regex = std :: regex (R"(\s*:\s*)");
std :: regex array_regex = std :: regex (R"(\s*,\s*)");
std :: regex vector_regex = std :: regex (R"(\s* \s*)");

std :: array < std :: string, 4 > NRRD_REQUIRED_FIELDS {"dimension", "type", "encoding", "sizes"};

struct DataSize{
public:
    DataSize(){
        m_dimensions = 0;
        m_segments = 0;
        m_slices = 0;
        m_width = 0;
        m_height = 0;
    }
    DataSize(unsigned int dimensions, unsigned int segments, unsigned int slices, unsigned int width, unsigned int height){
        m_dimensions = dimensions;
        m_segments = segments;
        m_slices = slices;
        m_width = width;
        m_height = height;
    }
    void set(unsigned int dimensions, unsigned int segments, unsigned int slices, unsigned int width, unsigned int height){
        m_dimensions = dimensions;
        m_segments = segments;
        m_slices = slices;
        m_width = width;
        m_height = height;
    }
    unsigned int m_dimensions;
    unsigned int m_width;
    unsigned int m_height;
    unsigned int m_slices;
    unsigned int m_segments;
};


enum{  _uchar = 0, _int16, _float
  }; // dtype;

enum{  _gzip = 0,  _bzip2, _raw, _txt
  }; // compression;

const std :: unordered_map < std :: string, int > dtype { {"unsigned char", _uchar},
                                                          {"unsigned int", _int16},
                                                          {"short", _int16},
                                                          {"int16", _int16},
                                                          {"float", _float}
                                                          };

const std :: unordered_map < std :: string, int > compress { {"gzip", _gzip},
                                                             {"gz", _gzip},
                                                             {"bzip2", _bzip2},
                                                             {"bz2", _bzip2},
                                                             {"raw", _raw},
                                                             {"txt", _txt},
                                                             {"text", _txt},
                                                             {"ascii", _txt}
                                                             };


template < class lambda = std :: function < std :: string (std :: string) > >
std :: vector < typename std :: result_of < lambda(std :: string) > :: type >
split (const std :: string & txt, const std :: regex & rgx, lambda func = [](std :: string s) -> std :: string { return s; })
{
  using type = typename std :: result_of < decltype(func)(std :: string) > :: type;

  std :: sregex_token_iterator beg(txt.begin(), txt.end(), rgx, -1);
  std :: sregex_token_iterator end;

  std :: size_t ntoken = std :: distance(beg, end);

  std :: vector < type > token(ntoken);
  std :: generate(token.begin(), token.end(),
                  [&] () mutable
                  {
                    return func(*(beg++));
                  });

  return token;
}


void check_magic_string (const std :: string & token)
{
  if (token[0] != 'N' || token[1] != 'R' || token[2] != 'R' || token[3] != 'D')
  {
    std :: cerr << "Invalid NRRD magic string. Found " << token << std :: endl;
    std :: exit(1);
  }
}

void check_header (std :: unordered_map < std :: string, std :: string > & header)
{
  // check required fields

  for ( auto const & key : NRRD_REQUIRED_FIELDS )
  {
    auto it = header.find(key);

    if ( it == header.end() )
    {
      std :: cerr << "Required field (" << key << ") not found" << std :: endl;
      std :: exit(1);
    }
  }

  // check number of dimensions equal to sizes
  const std :: string sizes = header.at("sizes");

  std :: size_t dims = std :: stoi(header.at("dimension"));
  std :: size_t size = std :: count(sizes.begin(), sizes.end(), ' ') + 1;

  if (dims != size)
  {
    std :: cerr << "Mismatch between dimensions and size" << std :: endl;
    std :: exit(1);
  }

}



template < typename dtype >
std :: vector < cv :: Mat > read_slices (std :: ifstream & is, const DataSize & data_size, const int & compression)
{
  const int buffer_size = data_size.m_segments * data_size.m_slices * data_size.m_width * data_size.m_height;

  std :: vector < dtype > data(buffer_size);

  // read the data buffer

  switch (compression)
  {
    case _gzip:
    {

      std :: vector < char > compressed;

      boost :: iostreams :: filtering_streambuf < boost :: iostreams :: input > in;
      in.push( boost :: iostreams :: gzip_decompressor() );
      in.push( is );

      boost :: iostreams :: copy(in, std :: back_inserter(compressed));
      boost :: iostreams :: close(in);

      // you need to reinterpret the buffer of decompressed data
      dtype * temp = (dtype*)compressed.data();
      std :: move(temp, temp + buffer_size, data.begin());
    } break;

    case _bzip2:
    {
      std :: vector < char > compressed;

      boost :: iostreams :: filtering_streambuf < boost :: iostreams :: input > in;
      in.push( boost :: iostreams :: bzip2_decompressor() );
      in.push( is );

      boost :: iostreams :: copy(in, std :: back_inserter(compressed));
      boost :: iostreams :: close(in);

      // you need to reinterpret the buffer of decompressed data
      dtype * temp = (dtype*)compressed.data();
      std :: move(temp, temp + buffer_size, data.begin());
    } break;

    case _raw:
    {
      is.read(reinterpret_cast < char* >(&data[0]), sizeof(dtype) * buffer_size);
    } break;

    case _txt:
    {
      for (int i = 0; i < buffer_size; ++i)
        is >> data[i];

    } break;

    default:
    break;
  }

  std :: vector < cv :: Mat > slices (data_size.m_slices);

  for (int i = 0; i < data_size.m_slices; ++i){
      slices[i] = cv :: Mat(data_size.m_width, data_size.m_height, CV_32SC1);
  }

  for (int s = 0 ; s < data_size.m_segments; s++){
      for (int i = 0; i < data_size.m_slices; ++i){
          int * img_data = (int*)(slices[i].data);
          for (int j = 0; j < data_size.m_width; ++j){
              for (int k = 0; k < data_size.m_height; ++k){
                  const int idx1 = j * data_size.m_height + k;
                  const int idx2 = k * data_size.m_width * data_size.m_slices * data_size.m_segments + j * data_size.m_slices * data_size.m_segments + i * data_size.m_segments + s;
//                  if (data[idx2] == 11){
//                      printf("Here bro %lu %lu %lu %lu\n", s, i, j, k);
//                  }
                  img_data[idx1] += data[idx2];
              }
          }
      }
  }

  return slices;
}



std :: unordered_map < std :: string, std :: string > read_header (std :: ifstream & is, std :: size_t & size)
{
  std :: unordered_map < std :: string, std :: string > header;
  std :: string row;

  std :: getline(is, row);
  check_magic_string(row);

  std :: getline(is, row); // read first token

  while ( !row.empty() )
  {

    // skip comments
    if (row[0] == '#')
    {
      std :: getline(is, row);
      continue;
    }

    auto token = split(row, value_regex);

    header[token[0]] = token[1];

    std :: getline(is, row);
  }

  check_header(header);

  size = is.tellg();

  return header;
}


std :: vector < cv :: Mat > read_nrrd (const std :: string & filename, const bool & print_header)
{
  std :: ifstream is(filename);

  if ( !is )
  {
    std :: cerr << "File not found. Given " << filename << std :: endl;
    std :: exit(1);
  }

  std :: size_t header_size = 0;
  auto header = read_header(is, header_size);

  if (print_header)
  {
    std :: cout << "header info:" << std :: endl;

    for ( auto const & [key, val] : header )
      std :: cout << "  \"" << key << "\" : " << val << std :: endl;
  }

  auto sizes = split(header.at("sizes"), vector_regex, [](const std :: string & x){return std :: stol(x);});
  long total_size = std :: accumulate(sizes.begin(), sizes.end(), 1, [](const auto & res, const auto & x){return res * x;});

  DataSize data_size;
  if (sizes.size() == 3){
      data_size.set(3, 1, sizes[0], sizes[1], sizes[2]);
  }
  else if (sizes.size() == 4){
      data_size.set(4, sizes[0], sizes[1], sizes[2], sizes[3]);
  }

  std :: size_t compression = compress.at(header.at("encoding"));

  std :: vector < cv :: Mat > slices (data_size.m_slices);

  const std :: string data_type = header.at("type");

  switch (dtype.at(data_type))
  {
    case _uchar:
      slices = read_slices < char >(is, data_size, compression);
    break;

    case _int16:
      slices = read_slices < short >(is, data_size, compression);
    break;

    case _float:
      slices = read_slices < float >(is, data_size, compression);
    break;

    default:
    {
      std :: cerr << "Data Type not yet supported. Please see http://teem.sourceforge.net/nrrd/format.html#Basic-Field-Specifications and manually add the right switch case" << std :: endl;
      std :: exit(1);
    } break;
  }

  is.close();

  return slices;
}


int main (int argc, char ** argv)
{

  std :: string filename = std :: string(argv[1]);

  auto slices = read_nrrd(filename, true);

  cv :: namedWindow("Test", cv :: WINDOW_NORMAL);

  std::cerr << "INFO! Number of Slices " << slices.size() << std::endl;
  for (std :: size_t i = 0; i < slices.size(); ++i){
    std::cerr << "\t INFO! Slice " << i << ": Number of Dims: " << slices[i].size() << std::endl;
  }
  

  for (std :: size_t i = 0; i < slices.size(); ++i)
  {
    auto s = slices[i];

    // to visualize a prettier image you should normalize it between [0, 255]
    cv :: normalize(s, s, 0, 255, cv :: NORM_MINMAX);
    // to visualize the images you must convert it into uchar type
    s.convertTo(s, CV_8UC1);

    cv :: imshow("Test", s);
    cv :: waitKey(10);
  }

  cv :: destroyWindow("Test");

  return 0;
}
