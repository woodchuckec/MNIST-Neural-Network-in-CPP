#include "Layer.h"
#include "Node.h"
#include <vector>
#include <iostream>
#include <cmath>
using namespace std;

Layer::Layer(std::vector<Node> _nodes, int _size, Layer *_prevLayer, Layer *_nextLayer) {
    nodes = _nodes;
    size = _size;
    prevLayer = _prevLayer;
    nextLayer = _nextLayer;
}

// Returns a reference to the node at index index
Node& Layer::operator[](int index) {
    return nodes[index];
}

MVector Layer::getActivations() {
    MVector acts = MVector(vector<double>(size), size);
    
    for(int i = 0; i < size; i++) {
        acts[i] = this->operator[](i).getActivation();
    }

    return acts;
}

void Layer::setActivations(MVector acts) {
    int n = acts.getSize();
    for(int i = 0; i < n; i++) {
        nodes[i].setActivation((double)acts[i] / 255.0);
    }
}