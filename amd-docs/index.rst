.. meta::
  :description: ComfyUI documentation
  :keywords: ComfyUI, ROCm, documentation, agent, GPU

.. _comfyui-documentation-index:

********************************************************************
ComfyUI on ROCm documentation
********************************************************************

While you can build workflows for generative AI tasks purely in code, the growing interest in GenAI
has led to increased demand for tools that don't require extensive programming knowledge.
ComfyUI provides you with a simple drag-and-drop interface for building GenAI workflows.
This guide will cover what ComfyUI is and how you can get it running on AMD Instinct GPUs.

`ComfyUI <https://docs.comfy.org/index.html>`__ is an open-source,
node-based interface for building and running image generation workflows with
diffusion models such as Stable Diffusion. Its modular graph-based design lets
you construct, customize, and share complex pipelines without writing code.

ComfyUI is part of the `ROCm-LLMExt toolkit
<https://rocm.docs.amd.com/projects/rocm-llmext/en/docs-26.04/>`__.

The ComfyUI public repository is located at `https://github.com/ROCm/ComfyUI/tree/release/0.18.2.amd0 <https://github.com/ROCm/ComfyUI/tree/release/0.18.2.amd0>`__.

.. grid:: 2
  :gutter: 3

  .. grid-item-card:: Install

    * :doc:`Install ComfyUI <install/comfyui-install>`

  .. grid-item-card:: How to

    * :doc:`Download and use models in ComfyUI <how-to/download-models>`
    * :doc:`Run the Hunyuan3D 2.1 template in ComfyUI <how-to/hunyuan3d-workflow>`
    * :doc:`Run the Wan 2.2 inference template in ComfyUI <how-to/wan-inference-workflow>`

  .. grid-item-card:: Reference

      * `Overview and reference documentation (upstream) <https://docs.comfy.org/>`__

To contribute to the documentation, refer to
`Contributing to ROCm <https://rocm.docs.amd.com/en/latest/contribute/contributing.html>`_.

You can find licensing information on the :doc:`Licensing <about/license>` page.
