
#include <algorithm>
#include <fstream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>
#include <cstdio>

#include "boost/scoped_ptr.hpp"
#include "boost/algorithm/string.hpp"
#include "gflags/gflags.h"
#include "glog/logging.h"
#include "leveldb/db.h"

#include "caffe/proto/caffe.pb.h"
#include "caffe/util/db.hpp"
// #include "caffe/util/format.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/rng.hpp"
#include "caffe/util/math_functions.hpp"

#include "opencv2/opencv.hpp"
#include "google/protobuf/text_format.h"
#include "stdint.h"
#include <iostream>
#include <cmath>

using namespace caffe;  // NOLINT(build/namespaces)
using std::pair;
using boost::scoped_ptr;
using namespace cv;
using namespace std;

DEFINE_bool(gray, false,
    "When this option is on, treat images as grayscale ones");
DEFINE_bool(shuffle, false,
    "Randomly shuffle the order of images and their labels");
DEFINE_string(backend, "lmdb",
        "The backend {lmdb, leveldb} for storing the result");
DEFINE_int32(resize_width, 0, "Width images are resized to");
DEFINE_int32(resize_height, 0, "Height images are resized to");
DEFINE_bool(check_size, false,
    "When this option is on, check that all the datum have the same size");
DEFINE_bool(encoded, false,
    "When this option is on, the encoded image will be save in datum");
DEFINE_string(encode_type, "",
    "Optional: What type should we encode the image as ('png','jpg',...).");
DEFINE_int32(channel, 3, "channel numbers of the image");

static bool ReadImageToMemory(const string &FileName, const int Height, const int Width, 
                        char *Pixels)
{
    //read image
    //cv::Mat OriginImage = cv::imread(FileName, cv::IMREAD_GRAYSCALE);
    cv::Mat OriginImage = cv::imread(FileName);     //3. read color image
    // LOG(INFO) << "filename = " << FileName << ".\n";
    CHECK(OriginImage.data) << "Failed to read the image.\n";

    //resize the image
    cv::Mat ResizeImage;
    cv::resize(OriginImage, ResizeImage, cv::Size(Width, Height));
    CHECK(ResizeImage.rows == Height) << "The heighs of Image is no equal to the input height.\n";
    CHECK(ResizeImage.cols == Width) << "The width of Image is no equal to the input width.\n";
    CHECK(ResizeImage.channels() == 3) << "The channel of Image is no equal to three.\n";    //4. should output the warning here

    // LOG(INFO) << "height " << ResizeImage.rows << " ";
    //LOG(INFO) << "weidth " << ResizeImage.cols << " ";
    //LOG(INFO) << "channels " << ResizeImage.channels() << "\n";

    // copy the image data to Pixels
    for (int HeightIndex = 0; HeightIndex < Height; ++HeightIndex)
    {
        const uchar* ptr = ResizeImage.ptr<uchar>(HeightIndex);
        int img_index = 0;
        for (int WidthIndex = 0; WidthIndex < Width; ++WidthIndex)
        {
            for (int ChannelIndex = 0; ChannelIndex < ResizeImage.channels(); ++ChannelIndex)
            {
                int datum_index = (ChannelIndex * Height + HeightIndex) * Width + WidthIndex;
                *(Pixels + datum_index) = static_cast<char>(ptr[img_index++]);
            }
        }
    }
    return true;
}

int main(int argc, char** argv) {
#ifdef USE_OPENCV
  ::google::InitGoogleLogging(argv[0]);
  // Print output to stderr (while still logging)
  FLAGS_alsologtostderr = 1;

#ifndef GFLAGS_GFLAGS_H_
  namespace gflags = google;
#endif

  gflags::SetUsageMessage("Convert a set of images to the leveldb/lmdb\n"
        "format used as input for Caffe.\n"
        "Usage:\n"
        "    convert_imageset [FLAGS] ROOTFOLDER/ LISTFILE DB_NAME\n"
        );
  gflags::ParseCommandLineFlags(&argc, &argv, true);

  if (argc < 4) {
    gflags::ShowUsageWithFlagsRestrict(argv[0], "tools/convert_siamese");
    return 1;
  }

  // const bool is_color = !FLAGS_gray;
  // const bool check_size = FLAGS_check_size;
  const bool encoded = FLAGS_encoded;
  const string encode_type = FLAGS_encode_type;

  std::ifstream infile(argv[2]);
  std::vector<std::pair<std::string, int> > lines;

  std::string filename;
  std::string filename2;

  
  int label;

  while (infile >> filename >> filename2 >> label) {
    string pairs = filename + ',' + filename2;
    lines.push_back(std::make_pair(pairs, label));
  }


  if (FLAGS_shuffle) {
    // randomly shuffle data
    LOG(INFO) << "Shuffling data";
    shuffle(lines.begin(), lines.end());
  }
  LOG(INFO) << "A total of " << lines.size() << " images.";

  if (encode_type.size() && !encoded)
    LOG(INFO) << "encode_type specified, assuming encoded=true.";

  int resize_height = std::max<int>(0, FLAGS_resize_height);
  int resize_width = std::max<int>(0, FLAGS_resize_width);
  // add channel info
  int channel = std::max<int>(1, FLAGS_channel);

  // 打开数据库
  // Open leveldb
  leveldb::DB* db;
  leveldb::Options options;
  options.create_if_missing = true;
  options.error_if_exists = true;
  leveldb::Status status = leveldb::DB::Open(
      options, argv[3], &db);
  CHECK(status.ok()) << "Failed to open leveldb " << argv[3]
        << ". Is it already existing?";

  // 保存到leveldb
  std::string root_folder(argv[1]);
  //char* Pixels = new char[2 * resize_height * resize_width];
  char* Pixels = new char[2 * resize_height * resize_width * channel];    //6. add channel
  const int kMaxKeyLength = 10;   //10
  char key[kMaxKeyLength];
  std::string value;

  // Storing to db
  caffe::Datum datum;
  //datum.set_channels(2);  // one channel for each image in the pair
  datum.set_channels(2 * channel);                //7. 3 channels for each image in the pair
  datum.set_height(resize_height);
  datum.set_width(resize_width);

  int count = 0;

  for (int line_id = 0; line_id < lines.size(); ++line_id) {
    // split pairs to two file name
    std::string pairs = lines[line_id].first;
    std::vector<std::string> names;

    boost::split(names, pairs, boost::is_any_of(","));
    
    char* FirstImagePixel = Pixels;
    // cout<<root_folder + lines[LineIndex].first<<endl;
    ReadImageToMemory(root_folder + names[0], resize_height, resize_width, FirstImagePixel);  //8. add channel here

    //char *SecondImagePixel = Pixels + resize_width * resize_height;
    char *SecondImagePixel = Pixels + resize_width * resize_height * channel;       //10. add channel
    ReadImageToMemory(root_folder + names[1], resize_height, resize_width, SecondImagePixel);  //9. add channel here

    // set image pair data
    // datum.set_data(Pixels, 2 * resize_height * resize_width);
    datum.set_data(Pixels, 2 * resize_height * resize_width * channel);     //11. correct
    datum.set_label(lines[line_id].second);

    // serialize datum to string
    datum.SerializeToString(&value);
    // int key_value = (int)(line_id);
    // _snprintf(key, kMaxKeyLength, "%08d", key_value);
    // sprintf(key, kMaxKeyLength, "%08d", key_value);
    // string keystr(key);
    // cout << "label: " << datum.label() << ' ' << "key index: " << keystr << endl;
    //sprintf_s(key, kMaxKeyLength, "%08d", LineIndex);     

    db->Put(leveldb::WriteOptions(), std::string(key), value);

    if (++count % 1000 == 0) {
      LOG(INFO) << "Processed " << count << " files.";
    }
  }

  delete db;
  delete[] Pixels;
  // write the last batch
  if (count % 1000 != 0) {
    LOG(INFO) << "Processed " << count << " files.";
  }
#else
  LOG(FATAL) << "This tool requires OpenCV; compile with USE_OPENCV.";
#endif  // USE_OPENCV
  return 0;
}
