#ifndef MVECTOR_H
#define MVECTOR_H

#include <vector>

class MVector {
private:
    std::vector<double> vector;
    int n;
public:
    MVector(std::vector<double> _vec = std::vector<double>(0), int _n = 0);
    int getSize() const {return n;};
    void setSize(int _n) {n = _n;};
    double& operator[](int index);
    void operator+=(MVector right);
    friend double operator*(MVector left, MVector right);
    friend MVector operator+(MVector left, MVector right);
};




#endif