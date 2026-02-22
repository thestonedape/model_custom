"""
BELT RESULTS COMPARISON SCRIPT
Compare your trained model results with BELT paper (Table VI)

Run this after training to evaluate replication quality.
"""

import sys
from pathlib import Path
import json

# Add to path
sys.path.insert(0, str(Path(__file__).parent))

# Color codes
GREEN = '\033[92m'
RED = '\033[91m'
YELLOW = '\033[93m'
BLUE = '\033[94m'
BOLD = '\033[1m'
RESET = '\033[0m'


def print_header(title):
    """Print section header"""
    print(f"\n{'='*80}")
    print(f"{BOLD}{title}{RESET}")
    print(f"{'='*80}\n")


def load_results(results_path):
    """Load results from JSON file"""
    try:
        with open(results_path, 'r') as f:
            return json.load(f)
    except FileNotFoundError:
        print(f"{RED}Error: Results file not found: {results_path}{RESET}")
        return None
    except json.JSONDecodeError:
        print(f"{RED}Error: Invalid JSON in results file{RESET}")
        return None


def compare_with_belt(your_results, model_name="Full BELT Model"):
    """Compare results with BELT paper Table VI"""
    
    print_header(f"COMPARING {model_name.upper()} WITH BELT PAPER")
    
    # BELT paper results (Table VI, page 3286)
    belt_results = {
        'EEG-to-Text': {
            'top10': 20.30,
            'description': 'EEG-to-Text encoder baseline'
        },
        'DeWave': {
            'top10': 21.98,
            'description': 'DeWave encoder baseline'
        },
        'D-Conformer (no VQ, no CL)': {
            'top10': 23.82,
            'description': 'D-Conformer only (ablation)'
        },
        'D-Conformer + VQ (no CL)': {
            'top10': 25.26,
            'description': 'With vector quantization, no contrastive'
        },
        'D-Conformer + VQ + CL (FULL)': {
            'top10': 31.04,
            'description': 'Full BELT model (target)'
        }
    }
    
    print(f"{BOLD}BELT Paper Results (Table VI):{RESET}")
    print("-" * 80)
    for model, data in belt_results.items():
        print(f"  {model:40s} {data['top10']:6.2f}%")
    
    print("\n" + "-" * 80)
    print(f"{BOLD}Your Results:{RESET}")
    print("-" * 80)
    
    # Extract your results
    if 'top10_accuracy' in your_results or 'top_10_accuracy' in your_results:
        your_top10 = your_results.get('top10_accuracy', 
                                     your_results.get('top_10_accuracy', 0))
        
        your_top5 = your_results.get('top5_accuracy', 
                                    your_results.get('top_5_accuracy', 0))
        
        your_top1 = your_results.get('top1_accuracy', 
                                    your_results.get('top_1_accuracy', 0))
        
        print(f"  Top-1 Accuracy:  {your_top1:6.2f}%")
        print(f"  Top-5 Accuracy:  {your_top5:6.2f}%")
        print(f"  Top-10 Accuracy: {your_top10:6.2f}%")
    else:
        print(f"{RED}  No top-k accuracy found in results{RESET}")
        return
    
    # Compare with BELT full model
    print("\n" + "=" * 80)
    print(f"{BOLD}COMPARISON WITH BELT FULL MODEL (31.04%):{RESET}")
    print("=" * 80)
    
    belt_target = 31.04
    difference = your_top10 - belt_target
    diff_percent = (difference / belt_target) * 100
    
    print(f"\n  BELT Result:     {belt_target:.2f}%")
    print(f"  Your Result:     {your_top10:.2f}%")
    print(f"  Difference:      {difference:+.2f}% ({diff_percent:+.1f}%)")
    
    # Evaluate replication quality
    print(f"\n{BOLD}Replication Quality:{RESET}\n")
    
    if abs(difference) < 1.0:
        print(f"  {GREEN}✓✓ EXCELLENT REPLICATION{RESET}")
        print(f"  Within 1% of BELT paper - Outstanding!")
        status = "EXCELLENT"
        
    elif abs(difference) < 2.0:
        print(f"  {GREEN}✓ GOOD REPLICATION{RESET}")
        print(f"  Within 2% of BELT paper - Very good!")
        status = "GOOD"
        
    elif abs(difference) < 5.0:
        print(f"  {YELLOW}⚠ ACCEPTABLE REPLICATION{RESET}")
        print(f"  Within 5% of BELT paper - Acceptable")
        print(f"  Consider minor tuning for better match")
        status = "ACCEPTABLE"
        
    else:
        print(f"  {RED}✗ POOR REPLICATION{RESET}")
        print(f"  >5% difference from BELT paper")
        print(f"  Review implementation carefully:")
        print(f"    - Check data pipeline (vocabulary, splits)")
        print(f"    - Verify architecture (6 blocks, VQ config)")
        print(f"    - Confirm hyperparameters match")
        print(f"    - Ensure all losses are computed correctly")
        status = "POOR"
    
    # Performance context
    print(f"\n{BOLD}Performance Context:{RESET}\n")
    
    baselines = [
        ('Random guess (1/500)', 0.20, 'random'),
        ('Random top-10 (10/500)', 2.00, 'random_top10'),
        ('EEG-to-Text baseline', 20.30, 'baseline'),
        ('DeWave baseline', 21.98, 'baseline'),
        ('D-Conformer (no VQ)', 23.82, 'ablation'),
        ('D-Conformer + VQ', 25.26, 'ablation'),
        ('BELT Full Model', 31.04, 'target'),
    ]
    
    print("  Benchmark                      Top-10      Your Result")
    print("  " + "-" * 76)
    
    for name, value, category in baselines:
        indicator = ""
        if category == 'target':
            color = GREEN if abs(your_top10 - value) < 2 else YELLOW
            indicator = f" {color}← TARGET{RESET}"
        elif your_top10 > value:
            indicator = f" {GREEN}✓ BEAT{RESET}"
        elif your_top10 < value:
            indicator = f" {RED}✗ BELOW{RESET}"
        
        print(f"  {name:30s} {value:6.2f}%     {your_top10:6.2f}% {indicator}")
    
    # Recommendations
    print(f"\n{BOLD}Recommendations:{RESET}\n")
    
    if status == "EXCELLENT" or status == "GOOD":
        print(f"  {GREEN}✓{RESET} Model replication successful!")
        print(f"  {GREEN}✓{RESET} Ready for further experiments")
        print(f"  {GREEN}✓{RESET} Can try enhancement techniques:")
        print(f"      - Label smoothing")
        print(f"      - MixUp augmentation")
        print(f"      - Better optimizers (AdamW)")
        print(f"      - Warmup schedules")
        
    elif status == "ACCEPTABLE":
        print(f"  {YELLOW}⚠{RESET} Replication acceptable but not perfect")
        print(f"  {YELLOW}⚠{RESET} Suggestions:")
        print(f"      1. Train for more epochs (try 80-100)")
        print(f"      2. Verify vocabulary has exactly 500 words")
        print(f"      3. Check data splits are 80/10/10")
        print(f"      4. Confirm VQ: K=1024, D=1024, β=0.3")
        print(f"      5. Verify loss weights: α=0.9, λ=1.0")
        
    else:
        print(f"  {RED}✗{RESET} Replication quality poor - investigation needed")
        print(f"  {RED}✗{RESET} Critical checks:")
        print(f"      1. Run: python verify_belt_implementation.py")
        print(f"      2. Verify data pipeline integrity")
        print(f"      3. Check all model components")
        print(f"      4. Confirm loss computation")
        print(f"      5. Review training logs for anomalies")
    
    # Additional metrics
    if 'test_loss' in your_results:
        print(f"\n{BOLD}Additional Metrics:{RESET}\n")
        print(f"  Test Loss:       {your_results['test_loss']:.4f}")
        
        if 'vq_perplexity' in your_results:
            perp = your_results['vq_perplexity']
            print(f"  VQ Perplexity:   {perp:.2f}")
            
            if 100 < perp < 500:
                print(f"    {GREEN}✓ Healthy codebook usage{RESET}")
            elif 50 < perp < 100:
                print(f"    {YELLOW}⚠ Low codebook usage{RESET}")
            elif perp < 50:
                print(f"    {RED}✗ Codebook collapse - check VQ implementation{RESET}")
    
    print("\n" + "=" * 80)
    
    return status


def print_ablation_comparison(ablation_results):
    """Compare ablation study with BELT paper"""
    
    print_header("ABLATION STUDY COMPARISON")
    
    belt_ablations = {
        'No VQ, No CL': 23.82,
        'With VQ, No CL': 25.26,
        'Full Model (VQ + CL)': 31.04
    }
    
    print(f"{BOLD}Expected Pattern (from BELT Table VI):{RESET}\n")
    print("  1. D-Conformer only:         23.82%")
    print("  2. + Vector Quantization:    25.26%  (+1.44%)")
    print("  3. + Contrastive Learning:   31.04%  (+5.78%)")
    print("\n  Total improvement: 7.22 percentage points\n")
    
    if ablation_results:
        print("-" * 80)
        print(f"{BOLD}Your Ablation Results:{RESET}\n")
        
        for config, top10 in ablation_results.items():
            belt_value = belt_ablations.get(config, None)
            if belt_value:
                diff = top10 - belt_value
                print(f"  {config:25s} {top10:6.2f}%  (BELT: {belt_value:.2f}%, diff: {diff:+.2f}%)")
            else:
                print(f"  {config:25s} {top10:6.2f}%")
    
    print("\n" + "=" * 80)


def main():
    """Main comparison function"""
    
    print_header("BELT RESULTS COMPARISON TOOL")
    
    print("This tool compares your trained model results with the BELT paper.")
    print("Provide path to your results JSON file.\n")
    
    # Check for default results paths
    default_paths = [
        "results/main_results/test_results.json",
        "results/full_belt/test_results.json",
        "results/test_results.json",
    ]
    
    results_path = None
    for path in default_paths:
        if Path(path).exists():
            results_path = path
            print(f"Found results at: {path}\n")
            break
    
    if not results_path:
        print("No results file found in default locations.")
        results_path = input("Enter path to results JSON file: ").strip()
    
    # Load results
    results = load_results(results_path)
    
    if not results:
        print(f"\n{RED}Could not load results. Exiting.{RESET}")
        return
    
    # Compare with BELT
    status = compare_with_belt(results)
    
    # Check for ablation results
    ablation_path = Path("results/ablation_results/test_results.json")
    if ablation_path.exists():
        ablation_results = load_results(str(ablation_path))
        if ablation_results:
            print_ablation_comparison({
                'No VQ, No CL': ablation_results.get('top10_accuracy', 0),
                'Full Model (VQ + CL)': results.get('top10_accuracy', 0)
            })
    
    # Final message
    print_header("SUMMARY")
    
    if status in ["EXCELLENT", "GOOD"]:
        print(f"{GREEN}✓ Your BELT implementation is working correctly!{RESET}\n")
        print("You have successfully replicated the BELT paper results.")
        print("Consider writing up your findings or trying enhancements.")
        
    elif status == "ACCEPTABLE":
        print(f"{YELLOW}⚠ Your implementation is on the right track.{RESET}\n")
        print("With minor adjustments, you should match BELT paper exactly.")
        print("Review the recommendations above.")
        
    else:
        print(f"{RED}✗ Implementation needs attention.{RESET}\n")
        print("Run comprehensive verification:")
        print("  python verify_belt_implementation.py")
    
    print("\n" + "=" * 80 + "\n")


if __name__ == "__main__":
    main()
