#!/bin/bash
# Shell script to run EMD-VB-CP experiments (for WSL/Linux)

echo "EMD-VB-CP Tensor Completion Experiments"
echo "========================================"

# Check if Python is available
if ! command -v python3 &> /dev/null; then
    echo "Error: Python 3 is not installed or not in PATH"
    echo "Please install Python 3.7+ and try again"
    exit 1
fi

# Install requirements
echo "Installing requirements..."
pip3 install -r requirements.txt
if [ $? -ne 0 ]; then
    echo "Error installing requirements"
    exit 1
fi

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
