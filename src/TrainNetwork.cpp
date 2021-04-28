#include "Network.h"
#include "TrainNetwork.h"
#include "MVector.h"
#include "Layer.h"
#include "InputLayer.h"
#include "HiddenLayer.h"
#include "PredictionLayer.h"
#include "Node.h"
#include <vector>
#include <cmath>
#include <iostream>
using namespace std;

// Helper mathematics functions are defined here
double sigmoidPrime(double x);
DataSet getDiagnosticTest(int num);
vector<vector<double>> transpose(vector<vector<double>> matrix);
vector<MVector> matrix_conv(vector<vector<double>> _matrix);
NetworkData scaleNetwork(NetworkData ntwk, double scale);
NetworkData sumNetworks(NetworkData left, NetworkData right);

TrainNetwork::TrainNetwork(vector<Layer> _layers) : Network(_layers) {}

Layer& TrainNetwork::operator[](int layerIndex) {
    return layers[layerIndex];
}

NetworkData TrainNetwork::getNetwork() {
    vector<vector<MVector>> weights = vector<vector<MVector>>(layers.size());
    // vector<vector<MVector>> weightsMV = vector<vector<MVector>>(layers.size());
    vector<vector<double>> biases = vector<vector<double>>(layers.size());

    for(int i = 0; i < layers.size(); i++) {
        weights[i] = vector<MVector>(layers[i].getSize());
        biases[i] = vector<double>(layers[i].getSize());

        

        for(int j = 0; j < layers[i].getSize(); j++) {

            weights[i][j] = layers[i][j].getWeights();
            biases[i][j] = layers[i][j].getBias();

        }

    }

    NetworkData nd = {weights, matrix_conv(biases)};
    //cout << "Network retrieved" << endl;
    return nd;
}

void TrainNetwork::feedforward(MVector image) {
    layers[0].setActivations(image);
    for(int layerInd = 1; layerInd < layers.size(); layerInd++) {
        MVector prevActs = layers[layerInd-1].getActivations();
        for(int nodeInd = 0; nodeInd < layers[layerInd].getSize(); nodeInd++) {
            layers[layerInd][nodeInd].update(prevActs);
        }
        //cout << endl;
    }
}

NetworkData TrainNetwork::backprop(MVector image, unsigned char label, double rate) {

    feedforward(image);
    // cout << endl << "Feedforward success" << endl;

    // gradient descent
    vector<vector<double>> errors = calcError(label);
    NetworkData adjustments = calcPartials(errors);

    // cout << endl << "calcPartials success" << endl;

    // scale the partials by the learning rate
    adjustments = scaleNetwork(adjustments, rate);

    return adjustments;
}

// I will be utterly shocked if any of this actually works - Ethan Wuitschick, 10:49 AM EST 4-25-21

void TrainNetwork::train(DataSet trainData, double rate) {
    // shortcut to easily generate a network of zeroes
    NetworkData adjustment = scaleNetwork(getNetwork(), 0);

    for(int i = 0; i < trainData.N; i++) {
        // add the scaled gradient from backpropogation to adjustment 
        adjustment = sumNetworks(adjustment, backprop(trainData.images[i], trainData.labels[i], rate));        
    }

    // average the adjustments over the total number of training images
    adjustment = scaleNetwork(adjustment, (-1.00 / (double)trainData.N));
    // update the network with adjustments
    setNetwork(sumNetworks(getNetwork(), adjustment));
    
}

void TrainNetwork::setNetwork(NetworkData network) {
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

// derivative of sigmoid function
double sigmoidPrime(double x) {
    double sig = 1 / ( 1 + pow(2.71828182845904, -x) );
    return (sig * (1 - sig));
}

vector<MVector> matrix_conv(vector<vector<double>> _matrix) {
    int cols = _matrix.size();
    vector<MVector> matrix = vector<MVector>(cols);

    for(int colInd = 0; colInd < cols; colInd++) matrix[colInd] = MVector(_matrix[colInd], _matrix[colInd].size());
    return matrix;
}

NetworkData sumNetworks(NetworkData left, NetworkData right) {
    NetworkData summed = left;

    //cout << "Starting sum loop";
    for(int i = 1; i < summed.weights.size(); i++) {
        for(int j = 0; j <summed.weights[i].size(); j++) {
            summed.biases[i][j] = left.biases[i][j] + right.biases[i][j];
            for(int k = 0; k < summed.weights[i][j].getSize(); k++) {
                summed.weights[i][j][k] = left.weights[i][j][k] + right.weights[i][j][k];
            }
        }
    }
    return summed;
}

NetworkData scaleNetwork(NetworkData ntwk, double scale) {
    NetworkData scaled = ntwk;

    // ignore input layer, no real relevant data
    for(int layerInd = 1; layerInd < ntwk.weights.size(); layerInd++) {
        for(int nodeInd = 0; nodeInd < ntwk.weights[layerInd].size(); nodeInd++) {
            scaled.biases[layerInd][nodeInd] *= scale;
            for(int weightInd = 0; weightInd < ntwk.weights[layerInd][nodeInd].getSize(); weightInd++) {
               scaled.weights[layerInd][nodeInd][weightInd] *= scale;
               
            }
            // cout << ".\t";
        }
    }
    // cout << endl << "Scale function ran successfully";
    // int b;
    // cin >> b;

    return scaled;
}

/*  calcError moves backwards through the network, calculating the error of a layer and using those values to
 *  calculate the previous layer's error.
 *  Each node in the network has an error, and the adjustments to be made are calculated from that error.*/
vector<vector<double>> TrainNetwork::calcError(unsigned char label) {
    int numLayers = layers.size();

    vector<vector<double>> errors = vector<vector<double>>(numLayers);

    // we backpropogate through the network to calculate the errors, no error exists for the input layer so we ignore it
    for(int layerInd = ( numLayers - 1 ); layerInd > 0; layerInd--) {
        errors[layerInd] = vector<double>(layers[layerInd].getSize()); // set the size of the current vector to be the size of that layer in the network
        
        // if we are in the output (prediction) layer, we use the label to find the error
        if(layerInd == (numLayers - 1)) {

            // loop that calculates the error for each prediction based on label
            for(int nodeInd = 0; nodeInd < errors[layerInd].size(); nodeInd++) {

                // if the current index matches label, the desired activation was 1
                if(nodeInd == (int)label) {
                    errors[layerInd][nodeInd] = (layers[layerInd][nodeInd].getActivation() - 1.0)*sigmoidPrime(layers[layerInd][nodeInd].getZ());
                } else { // otherwise the desired activation was 0
                    errors[layerInd][nodeInd] = (layers[layerInd][nodeInd].getActivation() - 0.0)*sigmoidPrime(layers[layerInd][nodeInd].getZ());
                }
            }
        } else { // now we backpropogate the output layer's errors
            
            /* For a given node in a previous layer, we use the sum of the products of the weight for that node by the error of the node that weight is in.
               That value is then multiplied by the derivative of the sigmoid function of the z of this node*/
            for(int nodeInd = 0; nodeInd < errors[layerInd].size(); nodeInd++) {
                double weightErrorSum = 0;
                for(int i = 0; i < errors[layerInd+1].size(); i++) {
                    weightErrorSum += layers[layerInd+1][i].getWeights()[nodeInd] * errors[layerInd+1][i];
                }
                errors[layerInd][nodeInd] = weightErrorSum * sigmoidPrime(layers[layerInd][nodeInd].getZ());
            }
        }
    }

    return errors;
}

/*  calcPartials uses the error for every node in the network to find the partial derivative for each weight and bias
 *  across the network. This function and calcError are the drivers for the backpropogation algorithm.*/
NetworkData TrainNetwork::calcPartials(vector<vector<double>> errors) {
    int numLayers = layers.size();

    vector<vector<double>> biasPartials = vector<vector<double>>(numLayers);
    vector<vector<vector<double>>> weightPartials = vector<vector<vector<double>>>(numLayers);

    for(int layerInd = 1; layerInd < numLayers; layerInd++) {
        biasPartials[layerInd] = vector<double>(layers[layerInd].getSize());
        weightPartials[layerInd] = vector<vector<double>>(layers[layerInd].getSize());

        for(int nodeInd = 0; nodeInd < biasPartials[layerInd].size(); nodeInd++) {
            weightPartials[layerInd][nodeInd] = vector<double>(layers[layerInd][nodeInd].getWeights().getSize());

            // calculate partials for node bias
            biasPartials[layerInd][nodeInd] = errors[layerInd][nodeInd];

            // calculate partials for node weight vector
            for(int weightInd = 0; weightInd < weightPartials[layerInd][nodeInd].size(); weightInd++) {
                weightPartials[layerInd][nodeInd][weightInd] = layers[layerInd-1][weightInd].getActivation() * errors[layerInd][nodeInd];
            }
        }
    }

    vector<vector<MVector>> weightPartialsMV = vector<vector<MVector>>(numLayers);
    for(int i = 0; i < weightPartialsMV.size(); i++)  {
        weightPartialsMV[i] = matrix_conv(weightPartials[i]);
    }
    vector<MVector> biasPartialsMV = matrix_conv(biasPartials);

    return {weightPartialsMV, biasPartialsMV};
}