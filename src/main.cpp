#include "MVector.h"
#include "Node.h"
#include "Layer.h"
#include "InputLayer.h"
#include "PredictionLayer.h"
#include "HiddenLayer.h"
#include "NetworkIO.h"
#include "TrainNetwork.h"
#include "TestNetwork.h"
#include <iostream>
#include <iomanip>
#include <vector>
#include <ctime>
#include <cstdlib>
#include <string>
#include <limits>

using namespace std;

char getCommand(string prompt, vector<char> valid);
int getInt(string prompt, int min, int max);
double getDouble(string prompt, double min, double max);

int main() {
    srand(time(0));
    NetworkIO io = NetworkIO();
    NetworkData ntwkData;

    char cmd;

    cout << "Convolutional Neural Network: Digit Recognition" << endl << endl << "START MENU:\n(n) Initialize New Network\n(l) Load network from save";
    cmd = getCommand("Enter command (n/l): ", {'n', 'l'});

    if(cmd == 'n') { 
        cout << endl << "Generating fresh network..." << endl;
        ntwkData = io.getRandomNetwork(5, {784, 16, 16, 16, 10});
    }
    else {
        cout << endl << "Loading network from networksav folder..." << endl;
        try {
            ntwkData = io.loadNetwork();
        } catch(const char *error) {
            cout << endl << error;
            return -1;
        }
    }

    // skip some lines
    cout << endl << endl;
    bool run = true;

    while(run) {
        TrainNetwork trainNet;
        TestNetwork testNet;
        DataSet training;
        DataSet testing;

        char choice;
        int i;

        cout << "MAIN MENU:\n(a) Train network\n(b) Test network\n(c) Save network\n(d) Exit";
        cmd = getCommand("Enter command: ", {'a', 'b', 'c', 'd', 'e'});
        switch(cmd) {
            case 'a': // run training
                int numEpochs;
                int numImages;
                double rate;
                trainNet = TrainNetwork();
                trainNet.setNetwork(ntwkData);
                if(io.getTrainingSet().N == 0) {
                    cout << "Loading training dataset. Please wait." << endl;
                    io.loadTraining();
                    cout << "Training dataset loaded." << endl;
                }
                numEpochs = getInt("Enter the number of epochs you would like to run (1 - 25000): ", 1, 25000);
                numImages = getInt("Enter the number of images per epoch you would like to run (1 - 250): ", 1, 250);
                rate = getDouble("Enter the learning rate for the training (0.0001 - 0.25): ", 0.0001, 0.25);
                training = io.getData(numImages, true);
                cout << endl << "Running training. Please wait.";
                for(int i = 0; i < numEpochs; i++) trainNet.train(training, rate);
                cout << endl << "Training finished.";
                ntwkData = trainNet.getNetwork();
                break;
            case 'b': //  run tests
                int numTests;
                testNet = TestNetwork();
                testNet.setNetwork(ntwkData);
                if(io.getTestingSet().N == 0) {
                    cout << "Loading testing dataset. Please wait." << endl;
                    io.loadTesting();
                    cout << "Training dataset loaded." << endl;
                }
                numTests = getInt("Enter the number of tests you would like to run (1-10000): ", 1, 10000);
                testing = io.getData(numTests, false);
                cout << "Running tests. Please wait." << endl;
                testNet.test(testing, true);
                break;
            case 'c':
                choice = getCommand("\nAre you sure you would like to save this network? It will override the current save.\nEnter command (y/n): ", {'y', 'n'});
                if(choice == 'y') {
                    io.saveNetwork(ntwkData);
                    cout << endl << "Network save successful." << endl;
                } else cout << endl << "Cancelling network save." << endl;
                break;
            case 'd':
                run = false;
                break;
            case 'e':
                testing = io.getData(1, false);
                testNet = TestNetwork();
                testNet.setNetwork(ntwkData);
                testNet.test(testing, true);
                for(i = 0; i < testNet[4].getSize(); i++) cout << testNet[4][i].getActivation() << " ";
                cout << endl;
                break;
            default:
                break;
        }
    }
    
}

char getCommand(string prompt, vector<char> valid) {
    char cmd = ' ';

    cout << endl << prompt;
    cin >> cmd;

    bool isValid = false;
    for(int i = 0; i < valid.size(); i++) if(cmd == valid[i]) isValid = true;

    if(isValid == false) {
        cout << endl << "Error: invalid command. Valid inputs are ";
        for(int i = 0; i < valid.size(); i++) {
            cout << valid[i];
            if(i != (valid.size()-1)) cout << ", ";
            else cout << "." << endl; 
        }
        cmd = getCommand(prompt, valid);
    }
    return cmd;
}


int getInt(string prompt, int min, int max) {
    int choice;
    cout << prompt;
    cin >> choice;
    while((cin.fail()) || (choice < min) || (choice > max)) {
        cout << endl << "Error: Please enter an integer between " << min << " and " << max << "." << endl;
        cin.clear();
        cin.ignore(numeric_limits<streamsize>::max(), '\n');
        cin >> choice;
    }
    return choice;
}

double getDouble(string prompt, double min, double max) {
    double choice;
    cout << prompt;
    cin >> choice;

    while((cin.fail()) || (choice < min) || (choice > max)) {
        cout << endl << "Error: please enter a number between " << min << " and " << max << "." << endl;
        cin.clear();
        cin.ignore(numeric_limits<streamsize>::max(), '\n');
        cin >> choice;
    }
    return choice;
}