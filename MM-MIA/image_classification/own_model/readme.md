
Images are resized to 64×64 and normalized to [0, 1].

## Model Architecture

The model is a compact CNN inspired by TinyVGG:

- Two convolutional blocks
- Each block: Conv → ReLU → Conv → ReLU → MaxPool
- Final classifier: Flatten → Linear

Output dimension matches the number of classes.

## Training Pipeline

1. Load dataset with transforms  
2. Initialize TinyVGG  
3. Train using Adam + CrossEntropy  
4. Evaluate on test set  
5. Save model weights  

Metrics tracked:
- Train loss / accuracy  
- Test loss / accuracy  

## Inference Pipeline

1. Load model weights  
2. Load and preprocess image  
3. Forward pass  
4. Softmax → predicted class  

## Improvements Possible

- Switch to PyTorch Lightning  
- Add data augmentation  
- Add early stopping  
- Use mixed precision  
- Try transfer learning (ResNet, EfficientNet)  