#include "MVector.h"
#include <vector> 
#include <iostream>
using namespace std;

MVector::MVector(std::vector<double> _vec, int _n) {
    //cout << "constructing mvector";
    vector = _vec;
    n = _n;
}

double& MVector::operator[](int index) {
    return vector[index];
}

void MVector::operator+=(MVector right) {
    if(n != right.getSize()) throw "ERROR: MVectors of differing sizes cannot be added.";
    else {
        for(int i = 0; i < n; i++) {
            vector[i] += right[i];
        }
    }
}

MVector operator+(MVector right, MVector left) {
    if(right.getSize() != left.getSize()) throw "ERROR: MVectors of differing sizes cannot be added.";
    else {
        int n = right.getSize();
        MVector added = MVector();
        added.setSize(n);

        for(int i = 0; i < n; i++) {
            added[i] = right[i] + left[i]; 
        }

        return added;
    }
}

double operator*(MVector right, MVector left) {
    if(right.getSize() != left.getSize()) throw "ERROR: MVectors of differing sizes cannot calculate a dot product.";
    else {
        int n = right.getSize();
        double product = 0;

        for(int i = 0; i < n; i++) {
            product += right[i] * left[i];
        }

        return product;
    }
}