# PLAN: brain tumor segmentation (LGG MRI). Tick each box only with evidence.
1. [x] Inspect data: count patients/slices, confirm every image has a mask, parse patient ID
       from folder names, report % of slices with empty masks, save outputs/samples.png
       (6 image+mask overlays). DONE WHEN: counts printed and overlays visibly aligned.
2. [x] Patient-level split 70/15/15 -> splits.json. DONE WHEN: assert no patient is in two sets.
3. [ ] Dataset + dataloader + augmentation (flips, small rotation, brightness), normalize.
       DONE WHEN: batch shapes printed, outputs/aug_preview.png looks sane.
4. [ ] Model: smp.Unet(resnet34, imagenet weights), Dice+BCE loss. Overfit test on 16 images.
       DONE WHEN: Dice > 0.9 on those 16 images.
5. [ ] train.py (args, mixed precision, best checkpoint by val Dice, curves png).
       Smoke-test with --max-samples 50. CHECKPOINT: stop, I run the full train on a GPU.
6. [ ] Evaluate on test set: per-patient Dice and IoU (mean +- std) -> RESULTS.md.
7. [ ] Error analysis: best/median/worst patient overlays, Dice vs tumor size plot, 3 failure modes.
8. [ ] Gradio demo: upload a slice -> mask overlay + tumor area.
9. [ ] README (problem, data, method, results table, figures, how to run, limitations,
       "research demo, not for clinical use"), requirements.txt, .gitignore data/ and checkpoints.
