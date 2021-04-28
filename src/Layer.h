#ifndef LAYER_H
#define LAYER_H

#include "Node.h"
#include <vector>

class Layer {
protected:
    std::vector<Node> nodes;
    int size;
    Layer *prevLayer;
    Layer *nextLayer;
public:
    Layer(std::vector<Node> _nodes = std::vector<Node>(0), int _size = 0, Layer *prev = nullptr, Layer *next = nullptr);
    
    std::vector<Node> getNodes() const { return nodes; };
    int getSize() const { return size; };
    void setNext(Layer *_next) {nextLayer = _next;};
    void setPrev(Layer *_prev) {prevLayer = _prev;};
    MVector getActivations();
    void setActivations(MVector activations);

    void setNodes(std::vector<Node> _nodes) { nodes = _nodes; };
    void setSize(int _size) { size = _size; };

    Node& operator[](int index);
};



#endif