from execute_model import run_model_single_parameter_node
from construct_model import get_model

model = get_model()
_, a, b, _ = run_model_single_parameter_node(model, [1, 1, 1, 1, 50, 0.1, 0.1])
print(a)
print(b)