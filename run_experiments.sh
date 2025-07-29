#!/bin/bash
# Shell script to run EMD-VB-CP experiments (for WSL/Linux)

echo "EMD-VB-CP Tensor Completion Experiments"
echo "========================================"

# Create results directory
mkdir -p results

# Run demo first
echo ""
echo "Running quick demo..."
cd experiments
python3 demo.py
if [ $? -ne 0 ]; then
    echo "Demo failed"
    exit 1
fi

# Ask user if they want to run full experiment
echo ""
read -p "Demo completed. Run full experiment? (y/N): " choice
case "$choice" in 
    y|Y ) 
        echo "Running full experiment..."
        python3 run_experiment.py
        if [ $? -ne 0 ]; then
            echo "Experiment failed"
            exit 1
        fi
        echo ""
        echo "Experiment completed! Check the results folder for outputs."
        ;;
    * ) 
        echo "Skipping full experiment."
        ;;
esac

cd ..
echo ""
echo "All done!"
