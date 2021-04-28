#ifndef TESTNETWORK_H
#define TESTNETWORK_H

#include "MVector.h"
#include "DataSet.h"
#include "NetworkData.h"
#include "Network.h"
#include <vector>

class TestNetwork : public Network {
public:
    TestNetwork(std::vector<Layer> _layers = std::vector<Layer>(0));
    void test(DataSet testData, bool printProgress);
    void feedforward(MVector image);
    short int getPrediction();
    Layer& operator[](int layerIndex) override;
    NetworkData getNetwork() override;
    void setNetwork(NetworkData network) override;
};



#endif