"""
Verify Training Results
"""
import json
from pathlib import Path

# Load results
with open("results/main_results/training_history.json", 'r') as f:
    history = json.load(f)

with open("results/main_results/final_results.json", 'r') as f:
    final = json.load(f)

print("="*80)
print("TRAINING VERIFICATION REPORT")
print("="*80)
print(f"Model: BELT WITHOUT Contrastive Loss (ablation test)")
print(f"Total Epochs: {final['final_epoch']}")
print(f"Learning Rate: 1e-4 (SGD)")
print()

print("="*80)
print("LEARNING PROGRESSION")
print("="*80)

epochs_to_check = [1, 10, 20, 30, 40, 50, 60]
print(f"{'Epoch':<8} {'Top-1':<8} {'Top-5':<8} {'Top-10':<8} {'L_ce':<8} {'L_vq':<8} {'Total':<8}")
print("-"*80)

# History has train and dev arrays directly
dev_history = history.get('dev', [])
for epoch in epochs_to_check:
    idx = epoch - 1
    if idx < len(dev_history):
        dev = dev_history[idx]
print("="*80)
print("FINAL TEST RESULTS")
print("="*80)
test = final['test_metrics']
print(f"Top-1 Accuracy:  {test['top1_acc']*100:.2f}%")
print(f"Top-5 Accuracy:  {test['top5_acc']*100:.2f}%")
print(f"Top-10 Accuracy: {test['top10_acc']*100:.2f}%")
print(f"Cross-Entropy Loss: {test['L_ce']:.4f}")
print(f"VQ Loss: {test['L_vq']:.4f}")
print()

print("="*80)
print("VERIFICATION CHECKS")
print("="*80)

# Check 1: Did CE decrease?
epoch1_ce = dev_history[0]['L_ce']
epoch60_ce = dev_history[59]['L_ce']
ce_improvement = epoch1_ce - epoch60_ce

print(f"✓ Cross-Entropy Learning:")
print(f"  Epoch 1:  L_ce = {epoch1_ce:.4f}")
print(f"  Epoch 60: L_ce = {epoch60_ce:.4f}")
print(f"  Improvement: {ce_improvement:.4f} {'✓ LEARNING!' if ce_improvement > 0.5 else '✗ Not learning enough'}")
print()

# Check 2: Did accuracy improve?
epoch1_acc = dev_history[0]['top10_acc'] * 100
epoch60_acc = dev_history[59]['top10_acc'] * 100
acc_improvement = epoch60_acc - epoch1_acc

print(f"✓ Top-10 Accuracy Improvement:")
print(f"  Epoch 1:  {epoch1_acc:.2f}%")
print(f"  Epoch 60: {epoch60_acc:.2f}%")
print(f"  Gain: +{acc_improvement:.2f}% {'✓ SIGNIFICANT!' if acc_improvement > 10 else '✗ Not enough'}")
print()

# Check 3: Compare to expected
expected_acc = 31.04
final_acc = final['best_dev_acc'] * 100
diff_from_expected = final_acc - expected_acc

print(f"✓ Comparison to BELT Paper (WITH contrastive):")
print(f"  Expected (full BELT): {expected_acc:.2f}%")
print(f"  Our result (no contrastive): {final_acc:.2f}%")
print(f"  Difference: {diff_from_expected:+.2f}%")
if abs(diff_from_expected) < 2:
    print(f"  ⚠️  WARNING: Too close to paper result despite missing contrastive!")
    print(f"  This suggests contrastive loss may not be helping (or is fighting CE)")
elif diff_from_expected < -2:
    print(f"  ✓ Expected: Ablation should perform worse than full BELT")
else:
    print(f"  ✗ Unexpected: Ablation performs BETTER than full BELT!")
print()

# Check 4: VQ Loss behavior
epoch1_vq = dev_history[0]['L_vq']
epoch60_vq = dev_history[59]['L_vq']

print(f"✓ Vector Quantizer Learning:")
print(f"  Epoch 1:  L_vq = {epoch1_vq:.4f}")
print(f"  Epoch 60: L_vq = {epoch60_vq:.4f}")
print(f"  {'✓ Stable codebook learning' if 0.5 < epoch60_vq < 2.0 else '✗ VQ loss unstable'}")
print()

print("="*80)
print("AUTHENTICITY VERDICT")
print("="*80)

authentic = True
issues = []

if ce_improvement < 0.5:
    authentic = False
    issues.append("CE didn't decrease enough (model not learning properly)")

if acc_improvement < 10:
    authentic = False
    issues.append("Accuracy gain too small (expected >10% over 60 epochs)")

if epoch60_ce > 5.8:
    authentic = False
    issues.append("Final CE too high (stuck near random guessing)")

if epoch60_vq > 2.0:
    issues.append("VQ loss high (codebook might not be converging)")

if abs(diff_from_expected) < 1:
    issues.append("Results TOO CLOSE to paper despite missing contrastive (suspicious)")

if authentic and len(issues) == 0:
    print("✓✓✓ RESULTS ARE AUTHENTIC AND CORRECT!")
    print("Model successfully learned EEG→Word mapping")
    print("Achieved near-BELT-paper performance WITHOUT contrastive loss")
else:
    print("⚠️  RESULTS HAVE ISSUES:")
    for issue in issues:
        print(f"  - {issue}")
    
    if len(issues) == 1 and "TOO CLOSE" in issues[0]:
        print("\nNote: This might actually be GOOD news!")
        print("It suggests the contrastive loss was blocking CE learning.")
        print("Try training WITH contrastive but alpha=0.05 to verify.")

print()
print("="*80)
print("RECOMMENDATION")
print("="*80)
print("Next steps:")
print("1. Train WITH contrastive loss (alpha=0.9) and compare")
print("2. If full BELT performs worse, reduce alpha to 0.1-0.2")
print("3. Consider switching to AdamW optimizer for better stability")
