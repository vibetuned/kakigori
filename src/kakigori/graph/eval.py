# pip install jiwer
import jiwer

def evaluate_humdrum_output(predicted_kern_list, ground_truth_kern_list):
    """
    predicted_kern_list: List of strings, where each string is a full system's **kern output.
    ground_truth_kern_list: List of strings, the ground truth equivalents.
    """
    
    # 1. SER (Token Error Rate)
    # Jiwer's 'wer' (Word Error Rate) splits by whitespace, which perfectly 
    # matches Humdrum tokens if they are separated by spaces or newlines.
    ser = jiwer.wer(ground_truth_kern_list, predicted_kern_list)
    
    # 2. CER (Character Error Rate)
    cer = jiwer.cer(ground_truth_kern_list, predicted_kern_list)
    
    # 3. LER (Line/System Error Rate)
    # This is essentially the percentage of completely failed systems.
    exact_matches = sum(1 for p, t in zip(predicted_kern_list, ground_truth_kern_list) if p.strip() == t.strip())
    ler = 1.0 - (exact_matches / len(ground_truth_kern_list))
    
    print(f"Token Error Rate (SER):      {ser * 100:.2f}%")
    print(f"Character Error Rate (CER):  {cer * 100:.2f}%")
    print(f"System Error Rate (LER):     {ler * 100:.2f}%")
    
    return ser, cer, ler