#!/usr/bin/env python3
"""
Corrected pipeline runner using ms_marco_eval.py
"""

import os
import sys
import json


def check_files():
    """Check all required files exist"""
    
    required_files = [
        'config.json',
        'llmlingua_compressor.py',
        'msmarco_evaluator.py',
        'evaluation/ms_marco_eval.py',  # CORRECTED: Main evaluation script
        'evaluation/rouge.py',
        'evaluation/bleu.py'
    ]
    
    missing = [f for f in required_files if not os.path.exists(f)]
    
    if missing:
        print("❌ Missing files:")
        for f in missing:
            print(f"  - {f}")
        
        if any('evaluation/' in f for f in missing):
            print(f"\n📥 Manual download required:")
            print(f"Go to: https://github.com/microsoft/MSMARCO-Question-Answering/tree/master/Evaluation")
            print(f"Download these files to evaluation/ directory:")
            print(f"  ✅ ms_marco_eval.py  (Main script - CORRECTED)")
            print(f"  ✅ rouge.py          (ROUGE dependency)")
            print(f"  ✅ bleu.py           (BLEU dependency)")
            print(f"  ⭕ run.sh            (Optional shell wrapper)")
        
        return False
    
    return True


def run_compression(num_examples=None):
    """Run compression phase"""
    
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    output_file = f'compression_results_{timestamp}.json'
    
    cmd = [sys.executable, 'llmlingua_compressor.py', '--output', output_file]
    if num_examples:
        cmd.extend(['--num_examples', str(num_examples)])
    
    print("="*50)
    print("COMPRESSION PHASE")
    print("="*50)
    print(f"Command: {' '.join(cmd)}")
    
    result = os.system(' '.join(cmd))
    if result != 0:
        print("❌ Compression failed")
        sys.exit(1)
    
    print(f"✅ Compression completed: {output_file}")
    return output_file


def run_evaluation(compression_file):
    """Run MS MARCO evaluation phase"""
    
    from datetime import datetime
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    eval_dir = f'msmarco_eval_{timestamp}'
    
    cmd = [
        sys.executable, 'msmarco_evaluator.py',
        '--results_file', compression_file,
        '--output_dir', eval_dir
    ]
    
    print("\n" + "="*50)
    print("MS MARCO EVALUATION PHASE (CORRECTED)")
    print("="*50)
    print(f"Using: evaluation/ms_marco_eval.py")
    print(f"Command: {' '.join(cmd)}")
    
    result = os.system(' '.join(cmd))
    if result != 0:
        print("❌ Evaluation failed")
        sys.exit(1)
    
    print(f"✅ Evaluation completed: {eval_dir}")
    return eval_dir


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Corrected LLMLingua2 + MS MARCO Pipeline')
    parser.add_argument('--phase', choices=['compression', 'evaluation', 'full'], default='full')
    parser.add_argument('--num_examples', type=int, help='Number of examples')
    parser.add_argument('--compression_results', help='Compression results file')
    
    args = parser.parse_args()
    
    # Check all files exist
    if not check_files():
        sys.exit(1)
    
    if args.phase == 'compression':
        output_file = run_compression(args.num_examples)
        print(f"\n🎉 Compression completed!")
        print(f"📁 Output: {output_file}")
        print(f"\nNext step:")
        print(f"python run_pipeline.py --phase evaluation --compression_results {output_file}")
    
    elif args.phase == 'evaluation':
        if not args.compression_results:
            print("❌ Need --compression_results for evaluation phase")
            sys.exit(1)
        
        if not os.path.exists(args.compression_results):
            print(f"❌ File not found: {args.compression_results}")
            sys.exit(1)
        
        eval_dir = run_evaluation(args.compression_results)
        print(f"\n🎉 Evaluation completed!")
        print(f"📁 Output: {eval_dir}")
    
    elif args.phase == 'full':
        print("🚀 Running full pipeline...")
        
        output_file = run_compression(args.num_examples)
        eval_dir = run_evaluation(output_file)
        
        print(f"\n🎉 Full pipeline completed!")
        print(f"📁 Compression: {output_file}")
        print(f"📁 Evaluation: {eval_dir}")
        print(f"\n📊 Check {eval_dir}/msmarco_results.json for scores")


if __name__ == '__main__':
    main()