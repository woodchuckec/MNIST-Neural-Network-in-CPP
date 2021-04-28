#include "MVector.h"
#include "Node.h"
#include <cmath>
#include <iostream>
using namespace std;

double sigmoid(double x) {
    return 1 / ( 1 + pow(2.71828182845904, -1 * x) );
}

Node::Node(MVector _weights, double _bias, double _z, double _activation) {
    weights = _weights;
    bias = _bias;
    z = _z;
    activation = _activation;
}

void Node::update(MVector activations) {
    z = (activations * weights) + bias;

    activation = sigmoid(z);
}