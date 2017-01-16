# C++ machine learning library 

This is a general neural network algorithm implementation
with functionality for an arbitrary number of layers and the
ability to specify arbitrary activition functions for any
of the non-linear units within the neural network.

I have primarily developed it for accomplishing the assignments
in my machine learning course at the University of Arkansas
under Dr. Michael Gashler. He provided guidance in the code
development. The code works with Dr. Gashler's Vec and Matrix
classes which interface nicely with data stored in the .arff
format.

It is has an object oriented design with two main classes
implementing the neural network: the neuralnet class and
its layer subclass. The neural net class inherits from the
supervisedlearner interface. There are other learning
algorithms such as simple linear regressors that also 
inherit from this interface.

There are filter classes which were not written by myself,
but have been modified to work with my implementations of
supervised learners. The filters are used to perform data
normalization and transformations from categorical representations
to one-hot or nominal representations.

There are still several loose ends in the code. For example,
right now the regressor class has not been updated to work
with the current filters.
