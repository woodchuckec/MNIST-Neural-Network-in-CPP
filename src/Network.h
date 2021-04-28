#ifndef NETWORK_H
#define NETWORK_H

#include "NetworkData.h"
#include "Layer.h"
#include <vector>

class Network {
protected:
    std::vector<Layer> layers;
public:
    Network(std::vector<Layer> _layers = std::vector<Layer>(0)) { layers = _layers; };
    virtual Layer& operator[](int index) = 0;
    virtual NetworkData getNetwork() = 0;
    virtual void setNetwork(NetworkData network) = 0;
};


#endif