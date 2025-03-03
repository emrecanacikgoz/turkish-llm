# **Evaluation**

## **1️⃣ Setup Environment**
For evaluation, set up a separate environment:

```bash
conda create --name llm_tr_eval python=3.8 -y
conda activate llm_tr_eval
```

Clone the evaluation framework:

```bash
git clone https://github.com/alisafaya/lm-evaluation-harness.git
cd lm-evaluation-harness
```

---

## **2️⃣ Run Evaluation**
Run the model evaluation on Turkish benchmarks:

```bash
lm_eval --model hf \
    --model_args pretrained=path/to/model,dtype="float" \
    --tasks arc_challenge_tr,truthfulqa_tr_mc1,truthfulqa_tr_mc2 \
    --device cuda:0 \
    --batch_size 1
```

🔹 **Parallelizing across multiple GPUs:**
```bash
lm_eval --model hf \
    --model_args pretrained=path/to/model,dtype="float",parallelize=True \
    --tasks arc_challenge_tr,truthfulqa_tr_mc1,truthfulqa_tr_mc2 \
    --device cuda:0 \
    --batch_size 1
```

📌 **Key Tasks:**
- **arc_challenge_tr**: Turkish version of the ARC challenge benchmark  
- **truthfulqa_tr_mc1**: Turkish multiple-choice factuality test  
- **truthfulqa_tr_mc2**: Another factuality multiple-choice benchmark  

---

## **📌 Notes**
- Ensure you replace `path/to/model` with the actual model path after fine-tuning.
- Parallelizing evaluation across available GPUs can speed up processing.
- The environment versions (`python=3.12` for training and `python=3.8` for evaluation) are chosen based on compatibility with respective libraries.

This should be a more structured, clear, and optimized version of your config. 🚀 Let me know if you need further refinements!