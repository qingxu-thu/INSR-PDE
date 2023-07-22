import os
from config import Config
import torch
print(torch.cuda.is_available(),torch.cuda.device_count())
# create experiment config containing all hyperparameters
cfg = Config("train")

# create model
if cfg.pde == "advection":
    from advection import Advection1DModel as neuralModel
elif cfg.pde == "fluid":
    from fluid import Fluid2DModel as neuralModel
elif cfg.pde == "flow_fluid":
    from flow_fluid import FlowFluid2DModel as neuralModel
elif cfg.pde == "elasticity":
    from elasticity import ElasticityModel as neuralModel
else:
    raise NotImplementedError
model = neuralModel(cfg)

output_folder = os.path.join(cfg.exp_dir, "results")
os.makedirs(output_folder, exist_ok=True)
print(output_folder)
# start time integration
for t in range(cfg.n_timesteps + 1):
    print(f"time step: {t}")
    if t == 0:
        model.initialize()
    else:
        model.step()

    model.write_output(output_folder)
