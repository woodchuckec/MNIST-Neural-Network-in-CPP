#ifndef NODE_H
#define NODE_H

#include "MVector.h"


class Node {
private:
    MVector weights;
    double bias;
    double z;
    double activation;
public:
    Node(MVector _weights = MVector(), double _bias = 0, double _z = 0, double _activation = 0);
    double getBias() const { return bias; };
    void setBias( double _bias) { bias = _bias; };
    MVector getWeights() const { return weights; };
    void setWeights(MVector _weights) { weights = _weights; };
    double getActivation() const { return activation; };
    double getZ() const { return z; };
    void setActivation( double _act ) { activation = _act; };
    void setZ( double _z ) { z = _z; };
    void update(MVector _acts);
};




#endif