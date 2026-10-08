# import onnx
# from onnx import helper, TensorProto

# # Load your YOLOv8 ONNX model
# model_path = '/media/sonnn/Data/model/vehicle_yolov8/best_640_op11.onnx'
# model = onnx.load(model_path)

# # Identify the output node you want to reshape
# output_node_name = 'output0'  # Replace with the actual name
# output_node = next(node for node in model.graph.output if node.name == output_node_name)

# # Create the Reshape node
# new_shape = [1, 8400, 11] # Your desired shape
# reshape_node = helper.make_node(
#     'Reshape',
#     inputs=[output_node.name, 'new_shape'],  # Input: original output, shape tensor
#     outputs=['reshaped_output'],
#     name='output_reshape'
# )

# # Create the shape tensor (constant input for the reshape)
# shape_tensor = helper.make_tensor(
#     name='new_shape',
#     data_type=TensorProto.FLOAT,
#     dims=[len(new_shape)],
#     vals=new_shape
# )
# model.graph.initializer.append(shape_tensor)

# # Add the Reshape node to the graph
# model.graph.node.append(reshape_node)

# # Update the model output
# output_node.name = 'reshaped_output'

# # Save the modified model
# onnx.save(model, '/media/sonnn/Data/model/vehicle_yolov8/best_640_op11_reshape.onnx')


import onnx
from onnx import TensorProto, helper

# Load your YOLOv8 ONNX model
model_path = "/media/sonnn/Data/model/vehicle_yolov8/best_640_op11.onnx"
model = onnx.load(model_path)

# Identify the output node you want to reshape
output_node_name = "output0"  # Replace with the actual name
output_node = next(node for node in model.graph.output if node.name == output_node_name)

# Create the Reshape node
new_shape = [1, 8400, 11]  # Your desired shape
reshape_node = helper.make_node(
    "Reshape",
    inputs=[output_node.name, "new_shape"],  # Input: original output, shape tensor
    outputs=["reshaped_output"],
    name="output_reshape",
)

# Create the shape tensor (constant input for the reshape)
shape_tensor = helper.make_tensor(name="new_shape", data_type=TensorProto.INT64, dims=[len(new_shape)], vals=new_shape)
model.graph.initializer.append(shape_tensor)

# Add the Reshape node to the graph
model.graph.node.append(reshape_node)

# Replace the original output with the reshaped output
output_node.name = "reshaped_output"
output_node.type.tensor_type.shape.dim[0].dim_value = new_shape[0]
output_node.type.tensor_type.shape.dim[1].dim_value = new_shape[1]
output_node.type.tensor_type.shape.dim[2].dim_value = new_shape[2]

# Save the modified model
onnx.save(model, "/media/sonnn/Data/model/vehicle_yolov8/best_640_op11_reshape.onnx")
