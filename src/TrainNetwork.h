#ifndef TRAINNETWORK_H
#define TRAINNETWORK_H

#include "Network.h"
#include "NetworkData.h"
#include "DataSet.h"
#include "MVector.h"
#include <vector>

class TrainNetwork : public Network {
public:
    TrainNetwork(std::vector<Layer> _layers = std::vector<Layer>(0));
    Layer& operator[](int layerIndex) override;
    NetworkData getNetwork() override;
    void train(DataSet trainData, double rate);
    NetworkData backprop(MVector image, unsigned char label, double rate);
    void feedforward(MVector image);
    std::vector<std::vector<double>> calcError(unsigned char label);
    NetworkData calcPartials(std::vector<std::vector<double>> errors);
    void setNetwork(NetworkData network) override;
};



#endif