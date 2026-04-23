.. meta::
  :description: ComfyUI example: Wan 2.2 inference workflow
  :keywords: ComfyUI, programming, agent, ROCm, example, sample, tutorial

.. _run-comfyui-wan-inference:

***************************************************************************
Wan 2.2 inference workflow
***************************************************************************

`Wan 2.2 <https://docs.comfy.org/tutorials/video/wan/wan2_2>`__ is an advanced AI
video generation model capable of producing high-resolution videos from text
descriptions or input images. With support for various aspect ratios and
customizable generation parameters, Wan 2.2 enables creative video synthesis
for a wide range of applications.

Setup
====================================================================

To run this workflow:

1. Establish an SSH tunnel between your local machine and the remote server:

   .. code-block:: bash

      ssh -N -L 8188:localhost:8188 <user>@<remote IP> -J <user>@<jump host IP>

2. Find the required models from the ComfyUI readme and download them to their respective folders under ``$COMFYUI_PATH/models`` on your server.

   For example, the following downloads text encoder safetensors (``umt5_xxl_fp8_e4m3fn_scaled.safetensors``) to the ``text_encoders`` folder on the ComfyUI remote server:

   .. code-block:: bash

      cd $COMFYUI_PATH/models
      cd text_encoders
      wget https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/text_encoders/umt5_xxl_fp8_e4m3fn_scaled.safetensors

3. Once the remote server is running, open the URL ``<remote IP>:8188`` (or ``localhost:8188``) in your local browser, unless you have changed the default port from 8188 to something else.

   Once greeted by the UI, navigate to **Templates → Video** and double-click the **Wan 2.2 14B Text to Video** template.

   .. figure:: ../images/wan22-templates-video.png
      :align: center
      :alt: ComfyUI Templates library with Video selected and the Wan 2.2 14B Text to Video card

      **Templates → Video:** under *Generation type*, choose **Video**, then open the **Wan 2.2 14B Text to Video** template (e.g. double-click the card).

Configuration
--------------------------------------------------------------------

Once the template is loaded, click **Run** to execute the workflow.

To trigger a run, enter prompts in the **CLIP Text Encode** nodes for positive and negative prompts and hit the blue **Run** button.

.. figure:: ../images/wan22-workflow-canvas.png
   :align: center
   :alt: Wan 2.2 workflow graph with CLIP Text Encode nodes and the Run control

   **Loaded workflow:** set positive and negative text in the **CLIP Text Encode** nodes, then use the blue **Run** button (top right). Model loaders and other nodes follow the template layout (e.g. **Wan2.2 T2V fp8_scaled**).

You can see workflow progress at the top of the UI in the green progress bar near the top of the browser.


Additional Resources
--------------------------------------------------------------------

- `ComfyUI Wan 2.2 Documentation <https://docs.comfy.org/tutorials/video/wan/wan2_2>`__
- `Wan 2.1 Model Files (HuggingFace) <https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged>`__


