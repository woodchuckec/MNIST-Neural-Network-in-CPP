#ifndef NETWORKDATA_H
#define NETWORKDATA_H

#include <vector>
#include "MVector.h"

struct NetworkData {
    std::vector<std::vector<MVector>> weights;
    std::vector<MVector> biases;
};

#endif