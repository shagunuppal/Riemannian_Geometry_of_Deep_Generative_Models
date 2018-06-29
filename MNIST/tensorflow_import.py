#import tensorflow
import onnx
#from onnx_tf.backend import prepare

model = onnx.load("MNIST.onnx")
onnx.checker.check_model(model)

print(onnx.helper.printable_graph(model.graph))
