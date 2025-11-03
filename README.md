# DDMCL: Meta-Path Diffusion Denoising and Multi-View Contrastive Learning for Recommendation

- This is the PyTorch implementation for DDMCL proposed in the paper DDMCL: Meta-Path Diffusion Denoising and Multi-View Contrastive Learning for Recommendation.

- The full source code will be released upon acceptance of this paper.

# Environment

- python 3.10.16
- pytorch 2.0.0
- numpy 1.26.4

# How to run
```
python Main.py --data amazon --ssl_reg 0.001 --ssl_reg1 0.005 --temp 0.2 --eps 0.5 --gnn_layer 4 --steps 50 --sampling_steps 12 --rebuild_k 20
```

