#!/bin/bash
# Quick Training Script for All BELT Models
# ============================================

echo "============================================"
echo "BELT Model Training Suite"
echo "============================================"
echo ""
echo "This script trains three variants:"
echo "  Model 1: BELT-Ablation (no bootstrapping)"
echo "  Model 2: BELT-Baseline (full BELT)"
echo "  Model 3: BELT-Enhanced (with improvements)"
echo ""
echo "Expected Performance:"
echo "  Model 1: ~25% top-10 accuracy"
echo "  Model 2: ~31% top-10 accuracy"
echo "  Model 3: ~37-39% top-10 accuracy"
echo "============================================"
echo ""

# Check if data is prepared
if [ ! -f "./dataset/vocabulary.pkl" ]; then
    echo "[ERROR] Dataset not prepared!"
    echo "Please run: python model_custom/prepare_data.py"
    exit 1
fi

# Ask which model to train
echo "Which model would you like to train?"
echo "  1) Model 1 - BELT Ablation (no bootstrapping)"
echo "  2) Model 2 - BELT Baseline (full BELT)"
echo "  3) Model 3 - BELT Enhanced (with improvements)"
echo "  4) All models (sequential)"
read -p "Enter choice [1-4]: " choice

case $choice in
    1)
        echo ""
        echo "Training Model 1: BELT-Ablation"
        echo "================================"
        python model_custom/experiments/model_without_bootstrapping.py \
            --config model_custom/config/belt_config.yaml \
            --mode train
        ;;
    2)
        echo ""
        echo "Training Model 2: BELT-Baseline"
        echo "================================"
        python model_custom/experiments/model_with_bootstrapping.py \
            --config model_custom/config/belt_config.yaml \
            --mode train
        ;;
    3)
        echo ""
        echo "Training Model 3: BELT-Enhanced"
        echo "================================"
        python model_custom/experiments/model_enhanced.py \
            --config model_custom/config/enhanced_config.yaml \
            --mode train
        ;;
    4)
        echo ""
        echo "Training ALL models sequentially"
        echo "================================"
        
        echo ""
        echo "[1/3] Training Model 1: BELT-Ablation"
        python model_custom/experiments/model_without_bootstrapping.py \
            --config model_custom/config/belt_config.yaml \
            --mode train
        
        echo ""
        echo "[2/3] Training Model 2: BELT-Baseline"
        python model_custom/experiments/model_with_bootstrapping.py \
            --config model_custom/config/belt_config.yaml \
            --mode train
        
        echo ""
        echo "[3/3] Training Model 3: BELT-Enhanced"
        python model_custom/experiments/model_enhanced.py \
            --config model_custom/config/enhanced_config.yaml \
            --mode train
        
        echo ""
        echo "All models trained! Results saved to:"
        echo "  - checkpoints_ablation/"
        echo "  - checkpoints/"
        echo "  - checkpoints_enhanced/"
        ;;
    *)
        echo "Invalid choice. Exiting."
        exit 1
        ;;
esac

echo ""
echo "Training complete!"
