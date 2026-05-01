.. meta::
  :description: Download models for ComfyUI
  :keywords: ComfyUI, programming, agent, ROCm, example, sample, tutorial

.. _comfyui-download-models:

********************************************************************
Download and use models in ComfyUI
********************************************************************

ComfyUI makes it easy to use models and build both simple and complex workflows with them.
However, you must first download the models you want to use. While you can download models directly from the UI,
the simplest and most stable method is to download them from a model repository (such as
`Hugging Face <https://huggingface.co/models>`__ or `Civitai <https://civitai.com/models>`__)
using command-line tools like ``curl``.

When downloading models through the command line, ensure you save them to the directories expected by ComfyUI:

* ``$COMFYUI_PATH/models/diffusion_models`` for diffusion models
* ``$COMFYUI_PATH/models/unet`` for UNet models
* ``$COMFYUI_PATH/models/vae`` for VAE models
* ``$COMFYUI_PATH/models/text_encoders`` for text encoders
* ``$COMFYUI_PATH/models/clip`` for CLIP models
* ``$COMFYUI_PATH/models/checkpoints`` for model checkpoints

Saving models to these directories makes it possible to use them in the standard ComfyUI nodes.

Download custom nodes
====================================================================

ComfyUI comes with a rich library of nodes and templates. Additionally, a vast community builds custom nodes that you can import and install in two ways: from the command line or using the ComfyUI Manager.

Command line installation
--------------------------------------------------------------------

To install nodes from the command line, clone a custom node repository into the ``custom_nodes`` directory and optionally install the dependencies:

.. code-block:: bash

   git clone <URL TO REPO> $COMFYUI_PATH/custom_nodes/<NAME OF NODE PACK>
   pip install -r $COMFYUI_PATH/custom_nodes/<NAME OF NODE PACK>/requirements.txt

.. note::

   For newly installed nodes to appear in the UI, ensure you restart the server.

ComfyUI Manager installation
--------------------------------------------------------------------

Installing nodes with the ComfyUI Manager is done directly in the UI. Select the custom nodes to install from the navigator and restart the server.