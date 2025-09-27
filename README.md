---
language:
- en
license: mit
task_categories:
- image-text-to-text
tags:
- Computer-Use
- Agent
---

<h1 style="
  font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',Helvetica,Arial,sans-serif;
  font-size:48px;
  font-weight:700;
  line-height:1.25;
  text-align:center;
  margin:0 0 24px;">
  OpenCUA: Open Foundations for Computer-Use Agents
</h1>

<div style="
  display:flex;
  justify-content:center;
  gap:12px;
  flex-wrap:wrap;
  margin-bottom:28px;">
  
  <a href="https://opencua.xlang.ai/" style="
     display:inline-block;
     padding:8px 24px;
     background:#2b2b2b;
     color:#ffffff;
     border-radius:36px;
     text-decoration:none;
     font-weight:600;
     font-size:16px;">
    🌐 Website
  </a>

  <a href="https://arxiv.org/abs/2508.09123" style="
     display:inline-block;
     padding:8px 24px;
     background:#2b2b2b;
     color:#ffffff;
     border-radius:36px;
     text-decoration:none;
     font-weight:600;
     font-size:16px;">
    📝 Paper
  </a>

  <a href="https://github.com/xlang-ai/OpenCUA" style="
     display:inline-block;
     padding:8px 24px;
     background:#2b2b2b;
     color:#ffffff;
     border-radius:36px;
     text-decoration:none;
     font-weight:600;
     font-size:16px;">
    💻 Code
  </a>
</div>

<div style="max-width:900px;margin:0 auto;">

<div style="text-align:center;">
  
# AgentNet Dataset

</div>

AgentNet is the first large-scale desktop computer-use agent trajectory dataset, containing 22.6K human-annotated computer-use tasks across Windows, macOS, and Ubuntu systems.

## Applications

This dataset enables training and evaluation of:
- Vision-language-action (VLA) models for computer use
- Multi-modal agents for desktop automation
- GUI understanding and interaction systems
- Cross-platform computer-use agents

## 🚀 Quick Start
Download the dataset here：
```
pip install -U huggingface_hub
huggingface-cli download xlangai/AgentNet --repo-type dataset --local-dir ./AgentNet
```

Use the following command to unzip the file (For exmaple, Ubuntu data):
```
cd path_to_your_zip_files

# Merge all the zips
zip -s 0 images.zip --out images-full.zip

# Unzip
unzip images-full.zip -d path_to_your_target_dir
```

## Action Space

The dataset uses PyAutoGUI actions and pre-defined agent related actions:

<div align="center">
<img src="https://cdn-uploads.huggingface.co/production/uploads/67b327cdd4665a0448eef7d5/FwA69rCLh81c-9CXaSE40.png" width="800" alt="AgentNet Action Space">
</div>

## Task Diversity


<div align="center">
  <img src="https://cdn-uploads.huggingface.co/production/uploads/67b327cdd4665a0448eef7d5/L281_EvnpQCeK9qShpqZX.png" width="400" alt="AgentNet Domain Distribution">
</div>

The dataset spans 4 main domains: Work (office tools, task management), Professional (creative design, development, data analysis, research), Daily (e-commerce, social media, entertainment), and System (configuration, web utilities). Tasks exhibit medium-high complexity with multi-application workflows, professional knowledge requirements, and uncommon feature usage.

<div style="text-align:center;">

## Data Synthesis Pipeline

</div>

Our data synthesis follows a 3-step process:

1.  **Tool Annotation** ([AgentNetTool](https://agentnet-tool.xlang.ai/)): Cross-platform annotation tool for capturing screen recordings, mouse/keyboard signals, and accessibility trees
2.  **Action Reduction & State-Action Matching** ([Code](https://github.com/xlang-ai/OpenCUA/tree/main/CoTGenerator)): Process raw demonstrations into compact state-action trajectories
3.  **CoT Synthesis** ([Code](https://github.com/xlang-ai/OpenCUA/tree/main/CoTGenerator)): Generate structured reasoning (Observation, Thought, Action) for each step using reflective long CoT framework

<div style="text-align:center;">

## Data Structure

</div>

Each JSONL file contains trajectories in the following structure:

```json
{
  "task_id": "20240927235321_5855063d-3f37-47a4-ab45-5247adfdb6f7",
  "instruction": "sort the table in ascending order based on the number column data in excel",
  "task_completed": false,
  "alignment_score": 7,
  "efficiency_score": 6, 
  "task_difficulty": 3,
  "natural_language_task": "Could you help me sort this table in Excel...",
  "actual_task": "Sort a table in WPS Office...",
  "traj": [
    {
      "index": 0,
      "image": "ea83c4aa-a4b1-48af-b439-0de7ee7b8d3f.png",
      "value": {
        "observation": "I'm looking at a WPS Office Excel spreadsheet...",
        "thought": "Since this is the first action...",
        "action": "Click on cell C2, which contains the number...",
        "code": "pyautogui.click(x=0.1632, y=0.2711)",
        "last_step_correct": true,
        "last_step_redundant": false,
        "reflection": "The action has successfully selected cell C2..."
      }
    }
  ]
}
```



## AgentNet Training Data Structure

### Data Components

**Original Annotation:**
- `instruction`: Human-annotated task description

**Synthesized by Summarizer:**
- `natural_language_task`: More natural task description
- `actual_task`: Detailed task specification
- `task_completed`, `alignment_score`, `efficiency_score`, `task_difficulty`: Task quality metrics

**Trajectory Steps (`traj`):**
Each step contains training components:
- `observation`: Generated visual scene description
- `thought`: Reasoning and planning process  
- `action`: Natural language action description
- `code`: Executable PyAutoGUI/function code

**Quality Control:**
- `last_step_correct`: Whether current step is correct
- `last_step_redundant`: Whether current step is redundant
- `reflection`: Generated by reflector for error analysis

### Training Message Format

During training, the data is converted into conversational format:

```python
# System Prompt
{"role": "system", "content": "You are a GUI agent..."}

# Multi-image History (previous steps)
{"role": "assistant", "content": "# Step 1
## Action:
Open Excel application
"}

{"role": "user", "image": "screenshot1.png"}
{"role": "assistant", "content": "# Step 2
## Action:
Click on File menu
"}

{"role": "user", "image": "screenshot2.png"}
{"role": "assistant", "content": "# Step 3
## Action:
Select data range
"}

# Current Step Instruction
{"role": "user", "image": "current_screenshot.png"}
{"role": "user", "content": "# Task Instruction:
sort the table in ascending order...

Please generate the next move..."}

# Target Response (L2 CoT example. Loss only applied to this part.)
{"role": "assistant", "content": "# Step 4
## Thought:
I need to select the data range first...
## Action:
Click on cell C2 to select the number column
## Code:
```python
pyautogui.click(x=0.1632, y=0.2711)
```"}
```

The training supports different CoT levels (L1: Action+Code, L2: Thought+Action+Code, L3: Observation+Thought+Action+Code) and action history.

See **Appendix G** for detailed training data examples with complete system prompts and multi-turn conversations.

<div style="text-align:center;">

## License

</div>

This project is licensed under the MIT License - see the LICENSE file for details.

<div style="text-align:center;">

## Research Use and Disclaimer

</div>

This dataset is intended for **research and educational purposes only**. 

### Prohibited Uses
- The dataset may **not** be used for any purpose or activity that violates applicable laws or regulations in any jurisdiction
- Use for illegal, unethical, or harmful activities is strictly prohibited
- **Any unauthorized reproduction, distribution, or use that infringes upon intellectual property rights is strictly forbidden**
- Users must respect privacy rights and confidentiality of any data subjects represented in the dataset

### Disclaimer
- The authors, contributors, and copyright holders are **not responsible** for any illegal, unethical, or harmful use of the dataset, nor for any direct or indirect damages resulting from such use
- Use of the "AgentNet" name, logo, or trademarks does **not** imply any endorsement or affiliation unless separate written permission is obtained
- Users are solely responsible for ensuring their use complies with applicable laws and regulations
- **Users assume all responsibility for verifying that their intended use does not violate any third-party rights or applicable laws**

<div style="text-align:center;">

## Citation

</div>

If you use OpenCUA in your research, please cite our work:

```bibtex
@misc{wang2025opencuaopenfoundationscomputeruse,
      title={OpenCUA: Open Foundations for Computer-Use Agents}, 
      author={Xinyuan Wang and Bowen Wang and Dunjie Lu and Junlin Yang and Tianbao Xie and Junli Wang and Jiaqi Deng and Xiaole Guo and Yiheng Xu and Chen Henry Wu and Zhennan Shen and Zhuokai Li and Ryan Li and Xiaochuan Li and Junda Chen and Boyuan Zheng and Peihang Li and Fangyu Lei and Ruisheng Cao and Yeqiao Fu and Dongchan Shin and Martin Shin and Jiarui Hu and Yuyan Wang and Jixuan Chen and Yuxiao Ye and Danyang Zhang and Dikang Du and Hao Hu and Huarong Chen and Zaida Zhou and Haotian Yao and Ziwei Chen and Qizheng Gu and Yipu Wang and Heng Wang and Diyi Yang and Victor Zhong and Flood Sung and Y. Charles and Zhilin Yang and Tao Yu},
      year={2025},
      eprint={2508.09123},
      archivePrefix={arXiv},
      primaryClass={cs.AI},
      url={https://arxiv.org/abs/2508.09123}, 
}
```

</div>