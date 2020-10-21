This is a FaaS-ish benchmark for serving various models using onnxruntime. It
isn't really ready for any particular FaaS system (yet) but it's a good
starting point. Models are taken from the onnx model zoo. Code was typically
modified from examples given in the onnx github repo.

# Models
## ferplus

## BERT\_squad
Adapted from: https://github.com/onnx/models/tree/master/text/machine_comprehension/bert-squad

Dependencies were taken verbatim, the contents of bertsquad.py was adapted from
the example ipython notebook.
