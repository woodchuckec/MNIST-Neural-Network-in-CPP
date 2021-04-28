#include "Network.h"
#include "TestNetwork.h"
#include "TrainNetwork.h"
#include "MVector.h"
#include "Layer.h"
#include "InputLayer.h"
#include "HiddenLayer.h"
#include "PredictionLayer.h"
#include "Node.h"
#include <vector>
#include <iostream>
#include <iomanip>

using namespace std;


TestNetwork::TestNetwork(vector<Layer> _layers) : Network(_layers) {}

// returns a reference to a layer in the network
Layer& TestNetwork::operator[](int layerIndex) {
    return layers[layerIndex];
}

void TestNetwork::setNetwork(NetworkData network) {
    int numLayers = network.weights.size();
    vector<Layer> newLayers = vector<Layer>(numLayers);

    //if(network.weights.size() > layers.size()) throw "Error in setNetwork(): network passed is too large.";

    for(int layerInd = 0; layerInd < numLayers; layerInd++) {
        vector<Node> layer = vector<Node>(network.weights[layerInd].size());
        // if this is the input layer
        if(layerInd==0) {
            layer = vector<Node>(784);
            for(int i = 0; i < layer.size(); i++) {
                MVector mv = MVector(vector<double>(0), 0);
                layer[i] = Node(mv, 0 ,0 ,0);
            }
            newLayers[layerInd] = InputLayer(layer, layer.size(), &newLayers[layerInd+1]);
        // this is a hidden or output layer
        } else {
            for(int i = 0; i < layer.size(); i++) layer[i] = Node(network.weights[layerInd][i], network.biases[layerInd][i], 0,0);
            // this is an output layer
            if(layerInd == (numLayers-1)) {
                newLayers[layerInd] = PredictionLayer(layer, layer.size(), &newLayers[layerInd-1]);
            // this is a hidden layer
            } else {
                newLayers[layerInd] = HiddenLayer(layer, layer.size(), &newLayers[layerInd-1], &newLayers[layerInd+1]);
            }
        }

        // for(int nodeInd = 0; nodeInd < network.weights[layerInd].size(); nodeInd++) {
        //     layers[layerInd][nodeInd].setWeights(network.weights[layerInd][nodeInd]);
        //     layers[layerInd][nodeInd].setBias(network.biases[layerInd][nodeInd]);
        // }
    }
    layers = newLayers;
}

// getNetwork() returns a NetworkData structure that holds the data for the weights and biases across the network
NetworkData TestNetwork::getNetwork() {
    vector<vector<MVector>> weights = vector<vector<MVector>>(layers.size());
    vector<vector<double>> biases = vector<vector<double>>(layers.size());

    for(int i = 0; i < layers.size(); i++) {
        // initialize each nested vector
        weights[i] = vector<MVector>(layers[i].getSize());
        biases[i] = vector<double>(layers[i].getSize());

        for(int j = 0; j < layers[i].getSize(); j++) {
            // fill the nested vectors with the network's data
            weights[i][j] = layers[i][j].getWeights();
            biases[i][j] = layers[i][j].getBias();
        }

    }
    
    vector<MVector> biasesMV = vector<MVector>(biases.size());
    for(int i = 0; i < biases.size(); i++) {
        biasesMV[i] = MVector(biases[i], biases[i].size());
    }

    // return a network data structure
    return {weights, biasesMV};
}

// feedforward feeds an image through the network, updating activations as it goes
void TestNetwork::feedforward(MVector image) {
    layers[0].setActivations(image);
    for(int layerInd = 1; layerInd < layers.size(); layerInd++) {
        MVector prevActs = layers[layerInd-1].getActivations();
        for(int nodeInd = 0; nodeInd < layers[layerInd].getSize(); nodeInd++) {
            layers[layerInd][nodeInd].update(prevActs);
        }
        //cout << endl;
    }
}

// getPrediction returns the index of the node in the prediction layer with the highest activation
short int TestNetwork::getPrediction() {
    // the prediction can be digit 0-9
    short int indexLargest = 0;
    int predictionLayer = layers.size()-1;
    // search for the largest activation in the predictionlayer
    for(short int ind = 0; ind < layers[predictionLayer].getSize(); ind++) 
        if(layers[predictionLayer][ind].getActivation() > layers[predictionLayer][indexLargest].getActivation()) indexLargest = ind;
    return indexLargest;
}

// this function drives tests, providing accuracy data and the ability to print the progress of the test
void TestNetwork::test(DataSet testData, bool printProgress) {
    cout << endl << "NETWORK TEST:" << endl;
    cout << "Number of images to test:\t" << testData.N << endl;

    int numSuccess = 0;
    for(int i = 0; i < testData.N; i++) {
        feedforward(testData.images[i]);
        if(getPrediction() == (int)testData.labels[i]) numSuccess++;
        if(printProgress) cout << "(" << (i+1) << "/" << testData.N << ") Prediction: " << getPrediction() << "\tCorrect Answer: " << (int)testData.labels[i] << "\tAccuracy so far: " << setprecision(4) << ((double)numSuccess / (double)i) << endl;
    }

    cout << endl << "Network TEST FINISHED" << endl;
    cout << "RESULTS:\tAccuracy: " << setprecision(5) << ( (double)numSuccess / (double)testData.N ) << endl;

}
