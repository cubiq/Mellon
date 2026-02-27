# Introduction

Modular Diffusers integration allows the use of the diffusers library with Mellon, at the moment, it is a proof of concept on how to integrate modular diffusers with a node graph system.

The key concepts are:

* **Dynamic nodes** — Instead of dozens of model-specific nodes, we have a small set of nodes that automatically adapt their interface based on the model you select. Learn them once, use them with any model.
* **Single-node workflows** — Thanks to Modular Diffusers' composable block system, you can collapse an entire pipeline into a single node. Run multiple workflows on the same canvas without the clutter.
* **Hub integration out of the box** — Custom blocks published to the Hugging Face Hub work instantly in Mellon. We provide a utility function to automatically generate the node interface from your block definition — no UI code required.

# Set Up Mellon

## Install Mellon

```bash
git clone https://github.com/cubiq/Mellon.git
cd Mellon
uv sync
uv run main.py
```

## Access UI from Local Browser

Go to: http://localhost:8088

You should now see the Mellon interface!

## Changing the port

If you just want to change the port to suit your needs, you can do it by creating a config.ini file in the root directory of Mellon and add the following:

```
[server]
host = 127.0.0.1
port = 8588
```

# Getting Started with Official Workflows

The fastest way to start is with our **official workflows** - pre-built templates ready to use.

In the left sidebar, click the **workflow icon** (the branching symbol), then expand **modular_diffusers**. You'll see several ready-to-use workflows:

* `text_to_image` - Generate images from text prompts
* `image_to_image` - Generate images with a prompt and a reference image
* `dynamic_node` - Load a node from Hub
* `multiple_image_edit` - Edit based on multiple images
* `quantization` - Run models with reduced memory using quantization

Simply **drag any workflow onto the canvas** to load it. The nodes come pre-connected, so you can just adjust your settings and run.

https://github.com/user-attachments/assets/a4d0604f-80ea-4470-80e6-53a73e584ca3

## The Basic  5-nodes Workflow

When you drag in `text_to_image` , you’ll see need these 5 nodes:

1. **Load Models** - This is the starting point. You can select your `Model Type` here and the rest of the workflow adapts their interface automatically. The outputs on the right (`Text Encoders`, `Denoise Model`, `VAE`, `Scheduler`, `Image Encoder`) connect via lines to downstream nodes - this is your data flow. Each component goes where it’s needed: e.g.  `Text Encoders` → `Encode Prompt` ,  `Denoise Model` + `Scheduler` → `Denoise` , and  `VAE` →  `Decode Latents`.

    **Our nodes are dynamic** - their interface changes based on the model you select. For example, Flux defaults to 28 steps with no negative prompt, Qwen supports negative prompts and defaults to 50 steps, while Z-Image-Turbo needs only 9 steps and doesn't use negative prompts. 

https://github.com/user-attachments/assets/4bbf74ac-404e-46bb-ae51-a84e65c25235

3. **Encode Prompt** - Encode your text prompt into embeddings.
4. **Denoise** - The denoise loop. Configure `Width`, `Height`, `Steps`, `Guidance Scale`, and `Seed` here.
5. **Decode Latents** - Decode latent output into an actual image using the VAE.
6. **Preview Image** - Displays your generated image.

That's all you need to know to get started. Type your prompt in the Encode Prompt node and hit **Run**:

https://github.com/user-attachments/assets/e563eeb0-4f9e-4a27-8304-49fd15b87550

## Switching Tasks with the Same Model

If your model supports multiple tasks, you don't need to load a separate workflow - just extend the one you have. For example, if you've already run text-to-image with Flux-Klein and want to do edit, simply add an Encode Image node and connect it to the Denoise node. The model stays loaded in memory, so there's no need to reload anything.

https://github.com/user-attachments/assets/ddbc3e06-6254-4595-8209-4cfd98d3aabc

## Single-Node Workflow

So far we've seen the **5-node workflow** - individual nodes for each stage of the pipeline connected together. But we also support a **single-node configuration** through the `Dynamic Block` node.

This is possible because of how Modular Diffusers works under the hood: pipeline stages are composable blocks that can be combined together however you want. You can split them across multiple nodes for fine-grained control, or combine everything into a single block with a custom UI - this helps to keep your canvas clean when you don't need the granularity.

With a single-node setup, you just need a `Dynamic Block` node and a repo id. Drag it onto the canvas, enter `YiYiXu/FLUX.2-klein-4B-modular`, type your prompt, and hit Run. That's it.

![Image](https://huggingface.co/datasets/OzzyGT/mellon_docs/resolve/main/demo4_singlenode.png)

This node can also connect to `Load Models` to share components with other nodes - we'll see this in action in the next section. Under the hood, components are managed efficiently: models are loaded once and stay in memory across runs, so hitting Run again with a different prompt doesn't reload anything. And when you connect multiple nodes to the same `Load Models`, they reuse the same loaded components.

## Combining Workflows

You can run multiple workflows on the same canvas, sharing model components between them. This isn't new - but our single-node really shines for this task, you don't need a canvas full of nodes and tangled connections. Just add another node, connect the shared components, and you're done.

In this example, we start with a single-node text-to-image workflow connected to `Load Models`. After generating an image, we drag in a second `Dynamic Block` node for image editing , connect it to the same `Load Models`, feed in the generated image with an editing prompt, and hit Run.

![Image](https://huggingface.co/datasets/OzzyGT/mellon_docs/resolve/main/demo5_2_workflow.png)

Only the second node runs, and it's fast - the model components are already loaded and shared so it is as efficient as the 5-node setup. 

# Use Custom Blocks from the Hub

Our system supports completely custom nodes - both the UI and backend code load dynamically from a Hugging Face repo. Custom block built with Modular Diffusers works out of the box in Mellon. 

Anyone can create and publish their own custom nodes to the Hub - just upload your block and it's instantly available for everyone to use in Mellon. You don't need to write UI code; we provide utility functions that automatically generate the Mellon configuration from your block definition.  see [Build Custom Block Guide] and [Using Custom Blocks with Mellon] for how to create and publish your own custom nodes.

In this example, we add a Gemini prompt expansion node (`diffusers/gemini-prompt-expander-mellon`) to an existing text-to-image workflow. Drag in a `Dynamic Block` node, enter the repo id, click Load, and it configures itself. Type a short prompt, connect its output to the Encode Prompt node, and run - Gemini expands your prompt into a detailed description before generating the image.

https://github.com/user-attachments/assets/d68bc8c1-1b1c-478a-b94b-1e498c60a4fc

# More Features

## Loading Models

You can load complete pipelines using the `Load Models` node or one model using the `Load Model` node to replace an already existing model from a pipeline, you can also choose if you want to load them from the HuggingFace Hub or from a local path.

![Image](https://huggingface.co/datasets/OzzyGT/mellon_docs/resolve/main/model.png)

## Quantization on the Fly

You can quantize models on the fly to reduce memory usage. Drag in a `Quantization Config`  node, input the model's `repo_id` and `subfolder`, and the node inspects the model architecture to dynamically populate a dropdown menu - letting you choose which layers to skip from quantization (for BnB). Connect it to the `Load Models` node and your model loads quantized

![Image](https://huggingface.co/datasets/OzzyGT/mellon_docs/resolve/main/quant.png)

## Using Loras

We have a Lora node where you can load a lora in the same way than any model, you just have to connect it to the lora input in the Models Loader node.

![Image](https://huggingface.co/datasets/OzzyGT/mellon_docs/resolve/main/lora.png)

## Pipeline Documentation

Each of our nodes maps to a Modular Diffusers pipeline on the backend, so all of them have a `Doc` output that lets you inspect the documentation of the underlying modular pipeline - connect it to a `Data Viewer` node to see what blocks are running, what inputs they accept, and how the pipeline is configured.

![Image](https://huggingface.co/datasets/OzzyGT/mellon_docs/resolve/main/doc.png)

## Comparing images

Mellon also has a built-in compare node you can use to see the changes in the images you generate, it’s as easy as changing the image preview for the image compare node and link the corresponding nodes. This is specially useful for testing and comparing the different guiders and techniques and see which one is better.

https://github.com/user-attachments/assets/058fcce6-28db-4faf-9063-b48a0a1ea592

