# TODOs


- [ ] what are my baselines: VILT vs uninitialized bert
	- [ ] how to use baseline that is not using fusion?
		- eg: fusion from hadamard product at the end... so only no coatts has still fusion. way to finetune without having to use the final concat/fusion
- [ ] adamw adjusted to the typical vilbert params: https://github.com/facebookresearch/vilbert-multi-task/blob/f22b84a9918a9aea2106e14ac1f6b32ad71492e3/train_concap.py#L465

- [ ] Statistical test: do cross-attention layers have significantly lower entropy (more focused)?
- [ ] attn entropy:
- Analyzing Multi-Head Self-Attention
- What Does BERT Look At?
- Is Attention Interpretable?
- Probing Multimodal Embeddings for Linguistic Properties

- [ ] gradient based attribution
- Grad-CAM: Visual Explanations from Deep Networks
- Generic Attention-model Explainability for Interpreting Bi-Modal Transformers
- Transformer Interpretability Beyond Attention Visualization"

- [ ] uinfy weighted loss for infonce in trainers
- [ ] tools to look into:
	- [ ] captum
	- [ ] alibi
	- [ ] bertviz
		```python
		from bertviz import head_view
		head_view(model, tokenizer, text_inputs, layer=4)  # visualize cross-attention
		```
- [ ] captum?
	- [ ] https://captum.ai/tutorials/Multimodal_VQA_Captum_Insights
- [ ] optuna:
	- [ ] run optuna for acc on both sets (mmimdb + hateful memes)
	- [ ] pruning
	- [ ] in ep_tracker: disable multiobjective or disable pruning

	- [ ] is my current setup even the right one?
		- [ ] optimize for alignment, not for loss


- [ ] arparse for experimenttracker: whenever I want to test alignment



- [x] implement experiment tracker
	- [ ] use test sets for alignment; no training on it. - currently on mmimdb, not on hm, still TODO!
	- [x] abstract class für trainer; hm, und mmimdb anpassen




- [ ] check if cka is right..
	- [ ] try with bigger bs for the data collection
- [x] self.fc outside of forward - refactor
- [x] add parameter how many samples to collect for visualization
	- [ ] more runs and avg out
- [ ] comparison of full-seq to cls.
	- [ ] training seemed to be more centered towards cls token alignment


- [ ] problem with contrastive term in pretraining: combined approach!



- [x] add dropout in attention
- [ ] caching , [mmap](https://github.com/DACUS1995/pytorch-mmap-dataset/blob/main/pytorch_mmap_dataset/dataset.py)

- [ ] is residual handling in crossattention correct?
- [ ] other datasets implement
	- [ ] find alignment datasets in literature

- [ ] data augmentation for AP pretraining
- [ ] implement further alignment measures
	- [x] cca
	- [ ] wasserstein distance
	- [x] svcca
	- [ ] sae (maybe)




## opts $\lor$ ideas
- [ ] investigating platonic representation hypothesis:
	- simply concat represetnations of bert + vit: use as baseline.
- [ ] pytorch hooks for intermediate layers
	- quite hard to implement, plus there is not much documentation on this topic.
- [ ] different batchsizes for tasks
	- maybe too difficult to implement!


## past TODOs
- [x] variable cka in analysis.
- [x] fix the alignment string to have same lenght
- [x] `src/evaluate.py` more flexible
	- [x] contrastive loss for other datasets; include in trainer
- [x] implement further datasets for alignment evaluation
	- [x] vqa
	- [x] mm-imbd
- [x] finetune bert and vit alone, without additional layers of vilbert
	- [x] abstr class for model, include all the heads.

- [x] unify hyperparam_optimizer and experiment_tracker.
- [x] is `num_samples=1000` still correct? should be controlled using GLOBAL VARS
- [x] better seeding
- [x] fix spelling issue in "costrative"
- [x] visualization of all the other measueres
	- [x] mknn
	- [x] jaccard - add to analysis
	- [x] rank - add to analysis
- [x] visualization of pretraining tasks - like acc, loss, etc
- [x] cosine scheduler
- [x] implement gradient accum.
- [x] use albuminations
- [x] easier dataset handling
- [x] add this to readme: `export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"`
- [x] visualization of cka, mutual knns
	- [x] implement a data collection pipeline
		- [ ] improve memory with `del`- in original pipeline=> better CKA estimations
- [x] mmimdb alignment vis
- [x] fix problem with ap pretraining only - has really bad performance, slightly worse than guessing!
	- 2025-08-23 22:35:30 - INFO  - trainer.py:train:691 - Epoch 4/4,
	```
	train loss MLM: 0.0000,
	test loss MLM: 10.5104,
	train loss AP: 0.6946,
	test loss AP: 0.6946,
	accuracy AP: 0.4986
	train loss MIM: 0.0000,
	test loss MIM: 8.8669
	```
- [x] Tokenizer for text dependency injected
- [x] pretrain dataset fix: filter out images that are not working
- [x] pretrain dataset mlm task
- [x] apparently there is a problem with the `transformers` library, where ViT implementation causes 10x-40x? https://x.com/jbohnslav/status/1950550831381782798, => own implementation of ViT (maybe adapt from dl VU, assignment 03)
- [x] fix problem with compile and saving

- [x] log everything
- [x] complete mim
    - [x] data augmentation pipeline.
    - [x] teacher, student ? this is to avoid moving target problem, but is it necessary? - not using this
    - [x] gradient stopping - not used, would require teacher-student setup

- [x] better config handling
- [x] infonce review

- [x] evaluate functino for measuring avg alignment measures.
- [x] complete pipeline for running experiments
- [x] hateful memes downsize to 224
- [x] unify the alignment measurements


- [x] self.query1 = nn.Linear(in_features=dim, out_features=dim) - moredimensions here
- [x] double and triple check if new architecture-fix is correct, because alignment visualizations look off
	- [x] problem with double forward pass for vit
- [x] upmc datatset
- [x] vqa
- [x] fix vilbert architecture
- [x] optuna- remove pruning, not necessary
	- [x] optuna rerun wit smaller lr range
	- [x] include easyvqa in optuna
- [x] run different configs, predetermined, so i can run several finetunes.
- [x] run from json files
	- [x] implement run from config
	- [x] test if impl is correct.
- [x] save everything that is necessary, is vi_biattention_ids, currently correct?
- [x] adapt finetune module to use experiment tracker.
- [x] proper naming for visualization of repr analyse
- [x] pretrainAP is wrong for my alignment analysis. half of the time it switches (like in the pretrain task). create separate class for analysis on conceptual captions.
	- [x] current workaround: probab in get_items is at 0
	-             if random.random() < 0.0:       # TODO: remove
- [x] mixed saving of intermediates: sometimes cls, sometimes full_seq
- [x] fix pretraining, several things are wrong
	- [x] compare ap contrastive with the contrastive in downstream
	- [x] pretraining problem with contrastive learning
	- [x] train loss is bigger then validation loss
		```
		2025-10-03 11:52:35 - INFO  - experiment_tracker.py:_run_pretrain:927 - Epoch 1/4,
		train loss MLM: 6.6594,
		test loss MLM: 4.7586,
		train loss AP: 0.6794,
		test loss AP: 0.6195,
		accuracy AP: 0.6492
		train loss MIM: 4.8037,
		test loss MIM: 4.1261
		```