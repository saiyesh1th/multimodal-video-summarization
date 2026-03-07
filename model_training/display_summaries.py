import os
import pandas as pd

def main():
    print("=========================================")
    print(" NLP PHASE 6: Summary Display")
    print("=========================================")
    
    csv_path = "./results/model_summaries.csv"
    
    if not os.path.exists(csv_path):
        print(f"[!] File not found: {csv_path}")
        print("[!] Please run Phase 5 first to generate the summaries.")
        return
        
    df = pd.read_csv(csv_path)
    
    if df.empty:
        print("[!] The summaries dataset is empty.")
        return
        
    print(f"[*] Displaying {len(df)} Model Summaries...\n")
    
    for idx, row in df.iterrows():
        print(f"[{idx+1}/{len(df)}] File: {row['file']}")
        print(f"Generated Summary: \n{row['generated_summary']}")
        print("-" * 50)
        
    print("\n[+] Phase 6 executed successfully.")

if __name__ == "__main__":
    main()
