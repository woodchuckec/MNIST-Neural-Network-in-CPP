Compilation:
1.	Once the code is downloaded, the first step is to extract /src/MNIST.zip. This compressed folder must have the four files contained inside extracted and placed in /src/. This is so that the program may properly load the databases.
2.	The next step is to compile the program. Open the command prompt on your pc and change the director to the folder containing the program. This is done with the command:
cd “directory to folder containing the program”
3.	Once the directory is set to the folder containing the program, the compile command must be run to generate an executable file:
g++ MVector.h MVector.cpp NetworkData.h DataSet.h NetworkIO.h NetworkIO.cpp Node.h Node.cpp Layer.h Layer.cpp InputLayer.h InputLayer.cpp HiddenLayer.h HiddenLayer.cpp PredictionLayer.h PredictionLayer.cpp Network.h TestNetwork.h TestNetwork.cpp TrainNetwork.h TrainNetwork.cpp main.cpp
4.	You can begin running the program by double clicking the a.exe file. An alternative to this is entering  start a.exe  while the console is still open.
5.	These steps only need to be ran once, as all that you need in the future is the a.exe file to run the program. Alternatively, the file downloaded from github can be used intead of compiling.

Usage:

Startup: Upon starting the program, the user is prompted to either choose to load the network from the save file or to have a new network generated. There is pre-loaded network data with the program that lives in the networksav folder, which can be directly interacted with by users in text files if they wish to. However, if the user wants to start fresh, they can enter n to create a new network. The program contains an 80% accurate network loaded in the save folder if users want to experience a trained network.

 
Main Menu: The menu shows after a network has been loaded and has four options:
	
	- (a) Train network: this puts the network in training mode. The network attempts to self-correct errors using the backpropagation algorithm while running epochs. The network adjusts itself after each epoch is finished running, and the user can choose the number of epochs and number of images per test.
	
	- (b) Test network: this puts the network in testing mode. The network runs a testing data set and keeps track of its success rate, printing the results of each test to the console.
	
	- (c) Save network: this saves the network data to the networksav folder. Beware, this overwrites the previous network that was saved.
	
	- (d) Exit: this terminates the program.

Tips: Because this program deals with large sets of data and heavy calculations over matrices, it may seem like the program is experiencing difficulties. The following are normal occurrences and do not mean it has crashed:
	
	- Upon launch, the first time a training or testing session is ran, the database is loaded. It may take some time to load the database, but do not panic if the program temporarily does not do anything.
	
	- When a large training session is run, the program can take up to a few minutes to train. During this time it will not print anything to the console, but do not exit. Instead let it finish training.
	
	- A network may have terrible accuracy, even after several training sessions. Training a neural network to accurately predict a digit can take hundreds of thousands of training examples, so do not expect good results after only a few minutes.

How To Train: Training the network can be a confusing process and, without patience, you may see poor results.

	- Training a fresh network: I recommend running 10,000 epochs at 200 images per epoch. Good learning rates here can range from 0.05 - 0.10. This first run took my network to 50% accurate, but results may vary.

	- Training an experienced network: I recommend running 25,000 epochs at 250 images per epoch. Good learning rates here can range from 0.005 – 0.01. One of these runs took my network from a 70% accuracy rate to a 75% accuracy rate.
