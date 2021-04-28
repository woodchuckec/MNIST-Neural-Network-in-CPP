#ifndef DATASET_H
#define DATASET_H

#include "MVector.h"
#include <vector>

struct DataSet {
    std::vector<MVector> images;
    std::vector<unsigned char> labels;
    unsigned short int N;
};




#endif