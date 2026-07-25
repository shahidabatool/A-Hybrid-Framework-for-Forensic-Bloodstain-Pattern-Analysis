#!/bin/bash
# Change directory to the folder where this script is located
cd "$(dirname "$0")"

# Print status message
echo "=================================================="
echo "🩸 Launching Forensic BPA & TSD Dashboard..."
echo "=================================================="

# Check if virtual environment exists
if [ -d "mrp_env" ]; then
    # Activate virtual environment
    source mrp_env/bin/activate
    # Run the streamlit dashboard
    streamlit run dashboard.py
else
    echo "[!] Error: 'mrp_env' virtual environment not found in this folder."
    echo "Please make sure you have created it first."
    read -p "Press Enter to exit..."
fi
