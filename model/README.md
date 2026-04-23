# Model

This folder contains the model implementations for the MCLDR project, a multi-view collaborative learning framework with denoising for recommendation systems.

## Files

- `model.py`: Implements the main MCLKR model that integrates denoising with collaborative filtering.
- `ViewLearner.py`: Defines the ViewLearner component for learning view-specific representations using MLP-based edge modeling.
- `denoise_encoder.py`: Contains the DenoiseEncoder class for denoising graph embeddings.
- `model_light_gcrec.py`: Lightweight implementation of the GCRec model.
- `denoising_amazon.py`: Denoising logic specific to the Amazon dataset.