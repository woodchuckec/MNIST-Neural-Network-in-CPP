#ifndef HIDDENLAYER_H
#define HIDDENLAYER_H

#include "Layer.h"
#include "Node.h"
#include <vector>
using std::vector;

class HiddenLayer : public Layer {
public:
    HiddenLayer(std::vector<Node> _nodes = std::vector<Node>(0), int _size = 0, Layer *_prevLayer = nullptr , Layer *_nextLayer = nullptr);
    Layer* getPrev() const { return prevLayer; };
    Layer* getNext() const { return nextLayer; };
};

#endif