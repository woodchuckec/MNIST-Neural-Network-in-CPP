/*
 *  Author: Ethan Wuitschick
 *  Description: This is the implementation file for the NetworkIO object defined in NetworkIO.h. The functions here manage the reading of datasets, the reading of saved network data, and the writing of network data to save it.
*/
#include "NetworkIO.h"
#include "DataSet.h"
#include "NetworkData.h"
#include "MVector.h"
#include <vector>
#include <fstream>
#include <iostream>
#include <ctime>
#include <cmath>
#include <string>
#include <sstream>
#include <algorithm>

using namespace std;

vector<vector<double>> readImages(unsigned short int numImages, bool training);
vector<double> readLabels(unsigned short int numImages, bool training);

/*  
 *  This function takes an integer in big endian and returns it in little endian.
 *  This code was originally written by user mrgloom on stackoverflow.com and was 
 *  used from this source because I had to search why the binary was not reading 
 *  correctly and found that I needed to implement this in order to read it on my architecture.
 *  Link: https://stackoverflow.com/questions/8286668/how-to-read-mnist-data-in-c
 */
int reverseInt (int i)
{
    unsigned char ch1, ch2, ch3, ch4;
    ch1=i&255;
    ch2=(i>>8)&255;
    ch3=(i>>16)&255;
    ch4=(i>>24)&255;
    return((int)ch1<<24)+((int)ch2<<16)+((int)ch3<<8)+ch4;
}

// CONSTRUCTOR for NetworkIO
NetworkIO::NetworkIO(bool testing, bool training) {
    if(testing) loadTesting();
    else Testing = {vector<MVector>(0), vector<unsigned char>(0), 0};
    if(training) loadTraining();
    else Training = {vector<MVector>(0), vector<unsigned char>(0), 0};
}

// SAVE AND LOAD NETWORK FUNCTIONS

/*  saveNetwork overwrites the files in the networksav file with the networkData struct passed to it.
 *  This function will throw a const char* if any of the files fail to open.
*/
void NetworkIO::saveNetwork(NetworkData ntwk) {
    int numLayers = ntwk.weights.size();

    // STAGE 1: Write the network configuration file
    // networkInfo is a file with key data about the network's dimensions. This data is useful in the loading process so that the program knows the dimensions to expect.
    ofstream networkInfo;
    string ntwkFname = networkSaveFolder;
    ntwkFname.append("/ntwkInfo.txt");
    // ntwkFname is networksav/ntwkInfo.txt
    networkInfo.open(ntwkFname, ios::trunc);
    // if the ntwkInfo file opened
    if(networkInfo) {
        networkInfo << numLayers << "\t";
        for(int i = 1; i < numLayers; i++) networkInfo << ntwk.weights[i].size() << "\t";
        networkInfo.close();
    } else {
        // if the file failed to open throw an error
        string error = "File open error: ";
        error.append(ntwkFname);
        error.append(" failed to open.");
        throw error;
    }
    
    // STAGE 2: Write the network weight files
    for(int i = 0; i < numLayers; i++) {
        // generate the filename to write to
        string fname = networkSaveFolder;
        fname.append("/layer");
        fname.append(to_string(i));
        fname.append(".txt");
        ofstream file;
        // open the file
        file.open(fname, ios::trunc);
        if(file) {
            
            for(int nodeInd = 0; nodeInd < ntwk.weights[i].size(); nodeInd++) {

                for(int weightInd = 0; weightInd < ntwk.weights[i][nodeInd].getSize(); weightInd++) {
                    file << ntwk.weights[i][nodeInd][weightInd] << "\t";
                }

                if(nodeInd != (ntwk.weights[i].size() - 1)) file << "\n";
            }

            file.close();
        } else {
            // if the file failed to open throw an error
            string error = "File open error: ";
            error.append(fname);
            error.append(" failed to open.");
            throw error;
        }

    }
    
    // STAGE 3: Write the network biases
    string fname = networkSaveFolder;
    fname.append("/biases.txt");
    // fname is now networksav/biases.txt
    ofstream file;
    file.open(fname, ios::trunc);
    // if file opened
    if(file) {
        for(int i = 1; i < numLayers; i++) {

            for(int nodeInd = 0; nodeInd < ntwk.biases[i].getSize(); nodeInd++) {
                file << ntwk.biases[i][nodeInd] << "\t";
            }
            if(i != (numLayers - 1)) file << "\n";
        }
        file.close();
    } else {
        // if the file failed to open throw an error
        string error = "File open error: ";
        error.append(fname);
        error.append(" failed to open.");
        throw error;
    }
}

/*  loadNetwork() loads a NetworkData structure from the folder networksav
 *  This function will throw a const char* if any of the necessary files fail to open, or if there is an error in the network config
 *  file.*/
NetworkData NetworkIO::loadNetwork() {
    
    // We first need to open the network configuration file that lays out the number of layers and the number of nodes in each layer

    // this lays out the basic fname that has the directory
    ifstream file;
    string fname = networkSaveFolder;
    fname.append("/");

    // this is the network's skeleton data that we read from the configuration file
    int numLayers = 0;
    vector<int> layerSizes;

    // STAGE 1: open network config file and load numLayers and layerSizes
    ifstream ntwkInfo;
    string ntwkFname = fname;
    ntwkFname.append("ntwkInfo.txt");
    ntwkInfo.open(ntwkFname);

    // if the open was a success, we read
    if(ntwkInfo) {
        // this string holds the data read from the tabulated file
        string in;

        // read until a tab
        getline(ntwkInfo, in, '\t');
        
        // create a stringstream out of in and store the integer to numLayers
        stringstream ss(in);
        ss >> numLayers; // ETHAN: WAS WRITING THE LOAD NETWORK, THIS FUNCTION WILL READ FROM A FORMATTING FILE TO KNOW HOW TO PROCESS OTHER FILES

        // initialize the layerSizes vector with a capacity equal to the number of layers
        layerSizes = vector<int>(numLayers);
        layerSizes[0] = 784;

        // iterate through the file reading the layer sizes
        for(int layerInd = 1; layerInd < numLayers; layerInd++) {
            getline(ntwkInfo, in, '\t');
            ss.clear();
            ss.str(in);
            ss >> layerSizes[layerInd];
        }
        ntwkInfo.close();
    } else throw "File open error: could not open the network configuration file.";

    // STAGE 2: iterate through layer files, reading the weights for the network

    vector<vector<MVector>> ntwkWeights = vector<vector<MVector>>(numLayers);
    vector<MVector> layerWeights;

    // this loop reads the layer text files into vectors of vector<double>
    for(int layerInd = 1; layerInd < numLayers; layerInd++) {

        layerWeights = vector<MVector>(layerSizes[layerInd]);

        string layerFname = fname;
        layerFname.append("layer");
        layerFname.append(to_string(layerInd));
        layerFname.append(".txt");
        file.open(layerFname, ios::in);

        for(int nodeInd = 0; nodeInd < layerSizes[layerInd]; nodeInd++) {
            string weightStr;
            getline(file, weightStr, '\n');

            vector<double> weights = vector<double>(layerSizes[layerInd-1]);

            // remove newline characters from the string
            weightStr.erase(remove(weightStr.begin(), weightStr.end(), '\n'), weightStr.end());

            stringstream ss(weightStr);
            for(int weightInd = 0; weightInd < layerSizes[layerInd-1]; weightInd++) {
                string weight;
                getline(ss, weight, '\t');
                stringstream wt(weight);
                wt >> weights[weightInd];
            }
            layerWeights[nodeInd] = MVector(weights, weights.size());

        }

        file.close();
        ntwkWeights[layerInd] = layerWeights;
    }

    // network's weights are loaded into a vector<vector<MVector>>

    vector<MVector> ntwkBiases = vector<MVector>(numLayers);
    // STAGE 3: network biases are loaded from biases.txt

    string biasFname = fname;
    biasFname.append("biases.txt");
    file.open(biasFname, ios::in);

    if(file) {
        for(int layerInd = 1; layerInd < numLayers; layerInd++) {
            string biasStr = "";
            getline(file, biasStr, '\n');

            // clear newline characters
            biasStr.erase(remove(biasStr.begin(), biasStr.end(), '\n'), biasStr.end());

            stringstream ss(biasStr);
            
            vector<double> layerBiases = vector<double>(layerSizes[layerInd]);

            for(int nodeInd = 0; nodeInd < layerSizes[layerInd]; nodeInd++) {
                string next;
                getline(ss, next, '\t');
                stringstream nxt(next);
                nxt >> layerBiases[nodeInd];
            }

            ntwkBiases[layerInd] = MVector(layerBiases, layerBiases.size());
        }

        file.close();
    } else {
        throw "File open error: could not open networksav/biases.txt";
    }

    return {ntwkWeights, ntwkBiases};
}

// Loads the testing DataSet
void NetworkIO::loadTesting() {

    vector<vector<double>> imgs = readImages(10000, false);
    vector<double> lbls = readLabels(10000, false);
    vector<MVector> imgsMV = vector<MVector>(10000);
    vector<unsigned char> ucharLabels = vector<unsigned char>(10000);

    for(int i = 0; i < imgs.size(); i++) {
        imgsMV[i] = MVector(imgs[i], 784);
        ucharLabels[i] = (unsigned char)lbls[i];
    }

    Testing = {imgsMV, ucharLabels, 10000};
}

// Loads the training DataSet
void NetworkIO::loadTraining() {

    vector<vector<double>> imgs = readImages(60000, true);
    vector<MVector> imgsMV = vector<MVector>(60000);
    vector<double> lbls = readLabels(60000, true);
    vector<unsigned char> ucharLabels = vector<unsigned char>(60000);

    for(int i = 0; i < imgs.size(); i++) {
        imgsMV[i] = MVector(imgs[i], 784);
        ucharLabels[i] = (unsigned char)lbls[i];
    }

    Training = {imgsMV, ucharLabels, 60000};
}

// Gets a subset of images and labels
DataSet NetworkIO::getData(short unsigned int numImages, bool training) {
    if(training) {
        if(Training.N < 60000) throw "Training dataset was loaded improperly.";
        vector<MVector> imgs = vector<MVector>(numImages);
        vector<unsigned char> lbls = vector<unsigned char>(numImages);
        int index;
        for(int i = 0; i < numImages; i++) {
            index = (rand() % 60000);
            imgs[i] = Training.images[index];
            lbls[i] = Training.labels[index];
        }
        
        DataSet ds = {imgs, lbls, numImages};
        return ds;

    } else {
        if(Testing.N < 10000) throw "Testing dataset was loaded improperly.";
        vector<MVector> imgs = vector<MVector>(numImages);
        vector<unsigned char> lbls = vector<unsigned char>(numImages);
        int index;
        for(int i = 0; i < numImages; i++) {
            index = (rand() % 10000);
            imgs[i] = Testing.images[index];
            lbls[i] = Testing.labels[index];

        }
        
        DataSet ds = {imgs, lbls, numImages};
        return ds;
    }
}

// Loads a vector<vector<double>> with image data from one of the database files
vector<vector<double>> readImages(unsigned short int numImages, bool training)
{
    // arr is a vector of vectors of doubles that will hold the information read from the file
    vector<vector<double>> arr;
    // set arr to have numImages number of elements of the type vector<double> with sizes 784
    arr.resize(numImages,vector<double>(784));

    // creates a file object to read the file. File is read in binary mode
    ifstream file; 
    if(training) file.open(train_data_fname, ios::binary);
    else file.open(test_data_fname, ios::binary);

    // if the file opens successfully
    if (file.is_open())
    {
        /* The MNIST database files are formatted at the start with tags that give information about the size of the database stored.
         * This data comes in the form of:
         * magicNum - data that is not useful
         * numberImages - the number of images whose data is included in the file
         * numRows - the number of rows / the number of vertical pixels in an image
         * numCols - the number of columns / the number of horizontal pixels in an image*/

        int magicNum=0;
        int numberImages=0;
        int numRows=0;
        int numCols=0;

        // read these tags. reverseInt is a method of adjusting the endianness; MNIST is written with big-endian in mind (most-significant byte is at smallest memory addr) but most PC architectures use little-endian (intel 64 and AMD 64).
        file.read((char*)&magicNum,sizeof(magicNum));
        magicNum= reverseInt(magicNum);
        file.read((char*)&numberImages,sizeof(numberImages));
        numberImages= reverseInt(numberImages);
        file.read((char*)&numRows,sizeof(numRows));
        numRows= reverseInt(numRows);
        file.read((char*)&numCols,sizeof(numCols));
        numCols= reverseInt(numCols);

        for(int i = 0; i < numberImages; ++i) {
            for(int r = 0; r < numRows; ++r)
            {
                for(int c=0; c < numCols; ++c)
                {
                    unsigned char temp = 0;
                    file.read( (char*)&temp, sizeof(temp) );
                    arr[i][(numRows*r) + c] = (double)temp;
                }
            }
        }
    } else {
        // if the file didn't open the database has either : not been extracted OR is not present
        if(training) throw "File open error: train-images.idx3-ubyte failed to open.";
        else throw "File open error: t10k-images.idx3-ubyte failed to open.";
    }
    
    file.close();
    return arr;
}

// Loads a vector<double> with label data from one of the database files
vector<double> readLabels(unsigned short int numImages, bool training)
{
    // arr is a vector of doubles that will contain the labels for each image stored
    vector<double> arr = vector<double>(numImages);

    // creates a file object to read from
    ifstream file; 
    if(training) file.open(train_label_fname, ios::binary);
    else file.open(test_label_fname, ios::binary);
    // if the file opens successfully
    if (file.is_open())
    {
        /* The MNIST database files are formatted at the start with tags that give information about the size of the database stored.
         * This data comes in the form of:
         * magicNum - data that is not useful
         * numberImages - the number of images whose data is included in the file*/
        int magicNum=0;
        int numberImages=0;
        // read magic number and number of images
        file.read( (char*)&magicNum, sizeof(magicNum) );
        magicNum = reverseInt(magicNum);
        file.read( (char*)&numberImages, sizeof(numberImages) );
        numberImages = reverseInt(numberImages);

        for(int i=0; i < numImages; ++i)
        {
            unsigned char temp=0;
            file.read((char*)&temp,sizeof(temp));
            arr[i] = (double)temp;
        }
    } else {
        // if the file didn't open the database has either : not been extracted OR is not present
        if(training) throw "File open error: train-labels.idx1-ubyte failed to open.";
        else throw "File open error: t10k-labels.idx1-ubyte failed to open.";
    }

    file.close();
    return arr;
}

/*  getRandomNetwork initializes a network based on the parameters of the function. The NetworkData struct returned will have numLayers layers and
 *  the layerSizes vector passed contains the size each layer should be.
*/
NetworkData NetworkIO::getRandomNetwork(int numLayers, vector<int> layerSizes) {
    NetworkData ntwk;

    // srand(time(nullptr));

    vector<vector<MVector>> weights = vector<vector<MVector>>(numLayers);
    vector<MVector> layerWeights;
    vector<MVector> biases = vector<MVector>(numLayers);

    // iterate through network layers to create
    for(int layerInd = 1; layerInd < numLayers; layerInd++) {

        layerWeights = vector<MVector>(layerSizes[layerInd]);


        for(int nodeInd = 0; nodeInd < layerSizes[layerInd]; nodeInd++) {
            int numWeights;
            if(layerInd == 0) numWeights = 0;
            else numWeights = layerSizes[layerInd - 1];

            vector<double> nodeWeights = vector<double>(numWeights);

            // the random element of this network is whether or not a generated weight is negative or positive
            for(int weightInd = 0; weightInd < numWeights; weightInd++){
                double val = sqrt(2) / sqrt(layerSizes[layerInd-1]);
                int num = (rand() % 3);
                if((num % 2) == 0) val *= -1;
                nodeWeights[weightInd] = val;
            }

            layerWeights[nodeInd] = MVector(nodeWeights, nodeWeights.size());  
        }

        weights[layerInd] = layerWeights;

    }

    // every bias is initialized to 0
    for(int layerInd = 0; layerInd < numLayers; layerInd++) {

        vector<double> layerBiases = vector<double>(layerSizes[layerInd]);

        for(int nodeInd = 0; nodeInd < layerBiases.size(); nodeInd++) layerBiases[nodeInd] = 0;

        biases[layerInd] = MVector(layerBiases, layerBiases.size());

    }

    return {weights, biases};

}