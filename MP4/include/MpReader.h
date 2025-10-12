#pragma once
#include <string>

class MP4Reader {
public:
    MP4Reader();
    bool read_file(const std::string& filename);
    // Add other public functions and data as needed
};
