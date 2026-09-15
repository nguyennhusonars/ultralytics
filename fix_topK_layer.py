import onnx

model_path = "/home/sonnn/Documents/ultralytics_sonnn/yolo26n-pose-320.onnx"
output_path = "/home/sonnn/Documents/ultralytics_sonnn/yolo26n-pose-320_fixed.onnx"

print(f"Loading {model_path} for deep attribute forcing...")
model = onnx.load(model_path)
graph = model.graph

fixed_count = 0

# Scan every node explicitly
for node in graph.node:
    if node.op_type == "TopK":
        print(f"\nFound Layer: '{node.name}'")
        
        # Check if an axis attribute already exists
        axis_attr = None
        for attr in node.attribute:
            if attr.name == "axis":
                axis_attr = attr
                break
        
        if axis_attr is not None:
            # An axis attribute exists; force it to positive 1
            print(f" -> Found existing axis attribute ({axis_attr.i}). Forcing to 1.")
            axis_attr.i = 1
            fixed_count += 1
        else:
            # The core problem: The attribute is missing, letting ONNX fall back to -1.
            # Create a brand new, explicit integer attribute for axis=1
            print(" -> No explicit axis attribute found (implicitly using -1). Injecting explicit axis=1.")
            new_attr = onnx.helper.make_attribute("axis", 1)
            node.attribute.append(new_attr)
            fixed_count += 1

# Save the thoroughly updated model structure
if fixed_count > 0:
    onnx.save(model, output_path)
    print(f"\n[SUCCESS] Successfully forced {fixed_count} TopK nodes to explicit positive axis paths.")
    print(f"Saved optimized, edge-friendly model to: '{output_path}'")
else:
    print("\nNo TopK nodes found.")
