# TDPO: Token-level Direct Preference Optimization

## What is this repo?

This repo includes a reference implementation of the TDPO algorithm for training language models from preference data, as described in the paper _Token-level Direct Preference Optimization_, which is built upon the implementation of [DPO](https://github.com/eric-mitchell/direct-preference-optimization). We have made partial modifications based on [DPO](https://github.com/eric-mitchell/direct-preference-optimization), and the specific usage method remains consistent with [DPO](https://github.com/eric-mitchell/direct-preference-optimization).

The code here supports any causal HuggingFace model- look at our examples in `config/model` to add your own. Adding your own datasets is also easy. See [the README section](https://github.com/huggingface/peft) on adding datasets.

The TDPO pipeline has two stages:

1. Run supervised fine-tuning (SFT) on the dataset(s) of interest.
2. Run preference learning on the model from step 1, using preference data (ideally from the same distribution as the SFT examples).

The files in this repo are:
- `train.py`: the main entry point for training (either SFT or TDPO preference-based training)
- `trainers.py`: the trainer classes (e.g., implementing the loop of learning as well as multi-GPU logic)
- `utils.py`: some convenience functions used by multiple other files
- `preference_datasets.py`: dataset processing logic for both SFT and TDPO preference-based training; **this is where you'll need to make some additions to train on your own data**



## A complete example

Let's work through a complete example training pythia 2.8B on the Anthropic-HH dataset.

### Step 1: Set up environment

    python3 -m venv env
    source env/bin/activate
    pip install -r requirements.txt



### Step 2: Run SFT

    python -u train.py model=pythia28 datasets=[hh] loss=sft exp_name=anthropic_tdpo_pythia28 gradient_accumulation_steps=2 batch_size=64 eval_batch_size=32 trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16



### Step 3: Run TDPO

For running **TDPO2**, we recommend the following command:

    python -u train.py model=pythia28 datasets=[hh] loss=tdpo loss.alpha=0.5 loss.beta=0.1 exp_name=anthropic_tdpo_pythia28 gradient_accumulation_steps=2 batch_size=64 eval_batch_size=32 trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16 model.archive=/path/to/archive/from/sft/LATEST/policy.pt



To run **TDPO1**, we only need to pass the additional parameter `loss.if_tdpo2=false`:

~~~
python -u train.py model=pythia28 datasets=[hh] loss=tdpo loss.alpha=0.5 loss.beta=0.1 loss.if_tdpo2=false exp_name=anthropic_tdpo_pythia28 gradient_accumulation_steps=2 batch_size=64 eval_batch_size=32 trainer=FSDPTrainer sample_during_eval=false model.fsdp_policy_mp=bfloat16 model.archive=/path/to/archive/from/sft/LATEST/policy.pt
~~~



We have provided wandb's training curve [here](https://wandb.ai/492277267/tdpo_demos).



## Acknowledgements

Many thanks to the contributors of [DPO](https://github.com/eric-mitchell/direct-preference-optimization) for their valuable contributions to the RLHF community. **For more detailed information, please refer to the  [DPO](https://github.com/eric-mitchell/direct-preference-optimization).**



## Citing TDPO

If TDPO or this repository is useful in your own research, you can use the following BibTeX entry:
