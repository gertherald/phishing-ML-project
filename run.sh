#!/bin/bash

################################################################################
# AIIP Batch 8 - Phishing Detection ML Pipeline
# Author: Gerald Chan Wei Heng
# Script: run.sh
# Purpose: Execute the complete ML pipeline
# Note: Dependencies will be installed automatically by GitHub Actions
#       DO NOT install dependencies in this script
################################################################################

echo "================================================================================"
echo "                   PHISHING DETECTION ML PIPELINE"
echo "                        AIIP Batch 8 Assessment"
echo "================================================================================"
echo ""
echo "Author: Gerald Chan Wei Heng"
echo "Date: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "================================================================================"
echo ""

# Color codes for better readability
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Function to print colored messages
print_success() {
    echo -e "${GREEN}✓ $1${NC}"
}

print_info() {
    echo -e "${BLUE}ℹ $1${NC}"
}

print_warning() {
    echo -e "${YELLOW}⚠ $1${NC}"
}

print_error() {
    echo -e "${RED}✗ $1${NC}"
}

# Function to print section headers
print_header() {
    echo ""
    echo "================================================================================"
    echo "$1"
    echo "================================================================================"
    echo ""
}

# Check if we're in the correct directory
print_header "STEP 1: VALIDATING ENVIRONMENT"

if [ ! -d "src" ]; then
    print_error "Error: 'src' directory not found!"
    echo "Please run this script from the project root directory."
    echo "Expected structure:"
    echo "  aiip8-GERALD_CHAN_WEIHENG_695B/"
    echo "  ├── run.sh (this script)"
    echo "  ├── src/"
    echo "  │   ├── data/"
    echo "  │   └── outputs/"
    echo "  ├── eda.ipynb"
    echo "  ├── README.md"
    echo "  └── requirements.txt"
    exit 1
fi
print_success "Project structure validated"

# Check if database exists
if [ ! -f "src/data/phishing.db" ]; then
    print_error "Error: Database file 'phishing.db' not found at src/data/"
    echo "Please ensure the database is located at: src/data/phishing.db"
    exit 1
fi
print_success "Database file found: src/data/phishing.db"

# Check Python availability
if ! command -v python &> /dev/null; then
    print_error "Python is not installed or not in PATH"
    exit 1
fi

python_version=$(python --version 2>&1)
print_success "Python detected: $python_version"

# Verify required Python files exist
print_info "Checking pipeline modules..."
required_files=(
    "src/config.py"
    "src/data_loader.py"
    "src/preprocessor.py"
    "src/trainer.py"
    "src/evaluator.py"
    "src/pipeline.py"
    "src/run.py"
)

all_files_exist=true
for file in "${required_files[@]}"; do
    if [ -f "$file" ]; then
        print_success "Found: $file"
    else
        print_error "Missing: $file"
        all_files_exist=false
    fi
done

if [ "$all_files_exist" = false ]; then
    print_error "Some required files are missing. Cannot proceed."
    exit 1
fi

# Check if requirements.txt exists
if [ -f "requirements.txt" ]; then
    print_success "requirements.txt found"
else
    print_warning "requirements.txt not found in root directory"
fi

# Display pipeline configuration
print_header "STEP 2: PIPELINE CONFIGURATION"
echo "Pipeline Strategy:"
echo "  • Feature Selection: Top 10 features only"
echo "  • Tree Models: Include DomainAge_Squared"
echo "  • Linear Models: Exclude DomainAge_Squared (avoid multicollinearity)"
echo "  • Outlier Handling: Feature-specific (protect phishing signatures)"
echo "  • Encoding: Target encoding for HostingProvider, One-hot for Industry"
echo ""
echo "Models to Train:"
echo "  1. Random Forest"
echo "  2. XGBoost"
echo "  3. Gradient Boosting"
echo "  4. Logistic Regression"
echo ""
echo "Expected Performance:"
echo "  • F1 Score: 0.93 - 0.96"
echo "  • Training Time: ~3-5 minutes"
echo "  • Output Features: ~18 (from 36-41 engineered)"

# Create outputs directory if it doesn't exist
print_header "STEP 3: PREPARING OUTPUT DIRECTORIES"
if [ ! -d "src/outputs" ]; then
    mkdir -p src/outputs
    print_success "Created: src/outputs/"
fi

if [ ! -d "src/outputs/models" ]; then
    mkdir -p src/outputs/models
    print_success "Created: src/outputs/models/"
fi

if [ ! -d "src/outputs/results" ]; then
    mkdir -p src/outputs/results
    print_success "Created: src/outputs/results/"
fi

# Execute the pipeline
print_header "STEP 4: EXECUTING ML PIPELINE"
print_info "Starting pipeline execution..."
echo ""

# Navigate to src directory and run the pipeline
cd src

# Run the Python pipeline
python run.py

# Capture exit code
exit_code=$?

# Return to root directory
cd ..

# Check if pipeline succeeded
print_header "STEP 5: PIPELINE EXECUTION SUMMARY"

if [ $exit_code -eq 0 ]; then
    print_success "Pipeline completed successfully!"
    echo ""
    echo "Output Locations:"
    echo "  • Trained Models: src/outputs/models/"
    echo "  • Evaluation Results: src/outputs/results/"
    echo "  • Execution Log: src/outputs/pipeline.log"
    echo ""
    
    # Check if key output files exist
    if [ -f "src/outputs/results/model_comparison.csv" ]; then
        print_success "Model comparison results generated"
        echo ""
        echo "Model Comparison:"
        echo "--------------------------------------------------------------------------------"
        cat src/outputs/results/model_comparison.csv
        echo "--------------------------------------------------------------------------------"
    fi
    
    echo ""
    print_info "To view detailed results, check:"
    echo "  • Confusion matrices: src/outputs/results/confusion_matrix_*.png"
    echo "  • ROC curves: src/outputs/results/roc_curve_*.png"
    echo "  • Feature importance: src/outputs/results/feature_importance_*.png"
    echo "  • Classification reports: src/outputs/results/classification_report_*.txt"
    echo ""
    
else
    print_error "Pipeline failed with exit code: $exit_code"
    echo ""
    echo "Troubleshooting Steps:"
    echo "  1. Check the log file: src/outputs/pipeline.log"
    echo "  2. Verify all dependencies are installed"
    echo "  3. Ensure database file is accessible"
    echo "  4. Check Python version compatibility (requires Python 3.8+)"
    echo ""
    exit $exit_code
fi

# Final summary
print_header "EXECUTION COMPLETE"
echo "Pipeline execution finished at: $(date '+%Y-%m-%d %H:%M:%S')"
echo ""
echo "Next Steps:"
echo "  1. Review model comparison in: src/outputs/results/model_comparison.csv"
echo "  2. Examine visualizations in: src/outputs/results/"
echo "  3. Check detailed logs in: src/outputs/pipeline.log"
echo "  4. Best model saved in: src/outputs/models/"
echo ""
print_success "Thank you for using the Phishing Detection ML Pipeline!"
echo "================================================================================"

exit $exit_code
