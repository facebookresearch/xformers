
# Examples

You can find in this folder a couple of ready to use examples, which include dataset, training loop and modelling through xFormers.
They are purposedly very simple, using what are nowadays toy datasets and small enough models to be trained on Colab or on a personal computer,
but they could easily be extended.

These examples are meant to illustrate some possible usecases for xFormers, but certainly not all of them.
In particular several of them use the `factory`, which is an optional interface to delegate the whole Transformer construction to xFormers.

In the future we'll also show how existing reference models (from [Pytorch image models](https://github.com/rwightman/pytorch-image-models) or Torchvision for instance)
can be patched with parts from xFormers which extent their capabilities.

## HOW TO

Please install the dependencies (`pip install -r requirements.txt`), after that all the examples can be run directly
(for instance `python3 microViT.py`). You should see a dataset being downloaded the first time, then training and the current loss,
and finally an inference example or some test loss and accuracy.

If your current machine does not expose enough RAM and the example reports an `OutOfMemoryError`, please adjust the batch size.
