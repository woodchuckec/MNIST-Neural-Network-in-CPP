#ifndef NETWORKIO_H
#define NETWORKIO_H

#include <string>
#include <vector>
#include "DataSet.h"
#include "NetworkData.h"

static std::string test_data_fname = "t10k-images.idx3-ubyte"; 
static std::string test_label_fname = "t10k-labels.idx1-ubyte";
static std::string train_data_fname = "train-images.idx3-ubyte";
static std::string train_label_fname = "train-labels.idx1-ubyte";
static std::string networkSaveFolder = "networksav";

class NetworkIO {
private:
    DataSet Training;
    DataSet Testing;
public:
    // These parameters allow users to choose which data set is loaded, if any, or both. They can be re-loaded later
    NetworkIO(bool testing = false, bool training = false);

    void loadTesting();
    void loadTraining();

    DataSet getData(short unsigned int numImages = 0, bool training = false);
    
    NetworkData loadNetwork();
    void saveNetwork(NetworkData network);

    NetworkData getRandomNetwork(int numLayers = 0, std::vector<int> layerSizes = std::vector<int>(0));
};


#endif