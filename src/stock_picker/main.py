#!/usr/bin/env python
import sys
import warnings

from datetime import datetime

from stock_picker.crew import StockPicker

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")

def run():
    """Run the stock picker crew"""
    inputs = {
        "sector": "AI"
    }
    result = StockPicker().stock_picker_crew().kickoff(inputs)

    print("\n\n=== FINAL RESULT ===\n\n")
    print(result.raw)

if __name__ == "__main__":
    run()


