### Usage:
Running parser.py converts a specified TVM graph into a kaas request, and then runs it. 
<br>
To run, use: `python parser.py [model]`, where model is the model you want to run. For example, to run the MNIST model, run `python parser.py mnist`. See setup to see how to setup a model.
<br>


### Setup: 
To setup a model, create a folder with the name of the model, and inside it create a file `parserUtils.py` with the following methods: <br>
1. `readData()` - gets the input  <br>
2. `getGraph()` - gets the graph json of the TVM model <br>
3. `getPath()` - gets the path to a file with all the functions used in the model <br>
4. `loadParams()` - get the constants in the model <br>
5. `getShapes()` - get the grid and block sizes for the cuda functions <br> 



Also, to see where the grid and block sizes come from, build the custom version of TVM for kaas. See https//github.com/NathanTP/tvm. 


Model specific setup:

MNIST: Run the Makefile inside the MNIST directory.



### Dependencies: 
Requires working installation of kaas, python-mnist, and TVM. The custom version of TVM for the project is suggested. 
