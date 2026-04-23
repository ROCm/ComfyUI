.. meta::
  :description: ComfyUI example: Hunyuan3D 2.1 workflow
  :keywords: ComfyUI, programming, agent, ROCm, example, sample, tutorial

.. _run-comfyui-hunyuan-3d:

********************************************************************
Hunyuan3D 2.1 workflow
********************************************************************

`Hunyuan3D <https://3d.hunyuanglobal.com/>`__ is an open-source 3D asset generation model released by Tencent,
capable of generating high-fidelity 3D models with high-resolution texture maps through text or images.

Hunyuan3D version 2.1 is Tencent's breakthrough 3D asset generation system that turns single images into
production-ready 3D models with physically-based rendering (PBR) materials. This release, Hunyuan3D-2.1,
includes improved texture quality with finer surface details and enhanced three-dimensional depth perception.

Setup
====================================================================

To run this workflow:

1. Follow the :ref:`comfyui-on-rocm-installation` steps to set up the ComfyUI environment and launch the ComfyUI WebUI.

2. Select **"Template"** from the navigation panel on the left and search for **Hunyuan 3D 2.1**.

   .. image:: ../images/hunyuan3d-workflow-01.png
      :alt: Template selection

3. Double-click on **Hunyuan3D 2.1** template.

4. Drag and drop an input image into the **"Step 2 - Upload image here"** node.
   
   Example image: https://comfyanonymous.github.io/ComfyUI_examples/hunyuan_image/hunyuan_image_example.png

   .. image:: ../images/hunyuan3d-workflow-02.png
      :alt: Upload image step

5. Click **"Run"**.

Configuration
--------------------------------------------------------------------

To modify input resolution:

- Edit the **"resolution"** option in the **Empty LaternHunyuan3Dv2** node.

Additional resources
--------------------------------------------------------------------

- `ComfyUI Hunyuan3D-2 Examples <https://comfyanonymous.github.io/ComfyUI_examples/hunyuan_image/>`__
- `Hunyuan 3D Models <https://3d-models.hunyuan.tencent.com/>`__


