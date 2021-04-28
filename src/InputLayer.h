#ifndef INPUTLAYER_H
#define INPUTLAYER_H

#include "Layer.h"
#include "Node.h"
#include "MVector.h"
#include <vector>

class InputLayer : public Layer {
public:
    InputLayer(std::vector<Node> _nodes = std::vector<Node>(0), int _size = 0, Layer *_nextLayer = nullptr);

    Layer* getNext() const { return nextLayer; };
};


#endif