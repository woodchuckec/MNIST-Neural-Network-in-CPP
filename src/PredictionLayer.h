#ifndef PREDICTIONLAYER_H
#define PREDICTIONLAYER_H

#include "Layer.h"

class PredictionLayer : public Layer {
public:
    PredictionLayer(std::vector<Node> _nodes = std::vector<Node>(0), int _size = 0, Layer *_prevLayer = nullptr);
    
    Layer* getPrev() const { return prevLayer; };
};

#endif