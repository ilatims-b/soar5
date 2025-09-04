#!/usr/bin/env python3
"""
Corrected MS MARCO evaluator using ms_marco_eval.py
"""

import json
import subprocess
import sys
import os
from typing import List, Dict


class CorrectMSMARCOEvaluator:
    """Wrapper for official MS MARCO ms_marco_eval.py script"""
    
    def __init__(self):
        self.eval_script = "evaluation/ms_marco_eval.py"
        
        # Check required files exist
        required_files = [
            "evaluation/ms_marco_eval.py",  # CORRECTED: Main evaluation script
            "evaluation/rouge.py", 
            "evaluation/bleu.py"
        ]
        
        missing = [f for f in required_files if not os.path.exists(f)]
        if missing:
            print("❌ Missing required files:")
            for f in missing:
                print(f"  - {f}")
            print(f"\n📥 Manual download required:")
            print(f"Go to: https://github.com/microsoft/MSMARCO-Question-Answering/tree/master/Evaluation")
            print(f"Download to evaluation/ directory:")
            print(f"  ✅ ms_marco_eval.py  (Main script)")
            print(f"  ✅ rouge.py          (ROUGE dependency)")  
            print(f"  ✅ bleu.py           (BLEU dependency)")
            print(f"  ⭕ run.sh            (Optional)")
            sys.exit(1)
    
    def format_predictions(self, results: List[Dict], method: str, output_file: str):
        """Format results for MS MARCO evaluation"""
        
        formatted_data = {}
        
        for result in results:
            query_id = str(result['query_id'])
            
            if method in result and 'response' in result[method]:
                response = result[method]['response'] or "No Answer Present."
            else:
                response = "No Answer Present."
            
            formatted_data[query_id] = response
        
        with open(output_file, 'w') as f:
            json.dump(formatted_data, f, indent=2)
        
        return output_file
    
    def format_references(self, results: List[Dict], output_file: str):
        """Format ground truth for MS MARCO evaluation"""
        
        references = {}
        
        for result in results:
            query_id = str(result['query_id'])
            ground_truth = result['ground_truth']
            references[query_id] = [ground_truth]  # MS MARCO expects list format
        
        with open(output_file, 'w') as f:
            json.dump(references, f, indent=2)
        
        return output_file
    
    def run_evaluation(self, references_file: str, predictions_file: str):
        """Run MS MARCO evaluation using ms_marco_eval.py"""
        
        cmd = [sys.executable, self.eval_script, references_file, predictions_file]
        
        print(f"Running: {' '.join(cmd)}")
        
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=True)
            
            # ms_marco_eval.py outputs metrics to stdout
            output = result.stdout.strip()
            print("MS MARCO Output:")
            print(output)
            
            # Parse metrics from output (may need adjustment based on actual output format)
            metrics = {}
            for line in output.split('\n'):
                if ':' in line and any(metric in line.lower() for metric in ['rouge', 'bleu', 'f1', 'exact']):
                    try:
                        parts = line.split(':')
                        metric_name = parts[0].strip()
                        metric_value = float(parts[1].strip())
                        metrics[metric_name.lower().replace('-', '_')] = metric_value
                    except:
                        pass
            
            return {'metrics': metrics, 'raw_output': output, 'success': True}
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Evaluation failed: {e}")
            print(f"STDERR: {e.stderr}")
            return {'error': str(e), 'stderr': e.stderr, 'success': False}
    
    def evaluate_all_methods(self, results: List[Dict], output_dir: str):
        """Evaluate all compression methods"""
        
        os.makedirs(output_dir, exist_ok=True)
        
        # Create references file (same for all methods)
        references_file = os.path.join(output_dir, 'references.json')
        self.format_references(results, references_file)
        
        methods = ['original', 'method1_rate', 'method2_target_tokens', 'method3_target_contexts']
        all_results = {}
        
        print(f"Running MS MARCO evaluation on {len(results)} examples...")
        print("="*60)
        
        for method in methods:
            print(f"\n📊 Evaluating {method}...")
            
            # Create predictions file for this method
            predictions_file = os.path.join(output_dir, f'predictions_{method}.json')
            self.format_predictions(results, method, predictions_file)
            
            # Run evaluation
            eval_result = self.run_evaluation(references_file, predictions_file)
            all_results[method] = eval_result
            
            if eval_result.get('success'):
                print(f"✅ {method} completed")
                if 'metrics' in eval_result and eval_result['metrics']:
                    for metric, value in eval_result['metrics'].items():
                        print(f"  {metric}: {value}")
            else:
                print(f"❌ {method} failed")
        
        # Save all results
        results_file = os.path.join(output_dir, 'msmarco_results.json')
        with open(results_file, 'w') as f:
            json.dump(all_results, f, indent=2)
        
        print(f"\n🎉 Evaluation completed!")
        print(f"📁 Results: {results_file}")
        print(f"📁 Files: {output_dir}/")
        
        # Summary table
        print(f"\n📊 RESULTS SUMMARY:")
        print("-" * 60)
        for method, result in all_results.items():
            if result.get('success') and 'metrics' in result:
                metrics_str = " | ".join([f"{k}:{v:.4f}" for k, v in result['metrics'].items()])
                print(f"{method:<25} {metrics_str}")
            else:
                print(f"{method:<25} FAILED")
        
        return all_results


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Corrected MS MARCO Evaluation')
    parser.add_argument('--results_file', required=True, help='Compression results JSON')
    parser.add_argument('--output_dir', default='msmarco_eval', help='Output directory')
    
    args = parser.parse_args()
    
    # Load results
    with open(args.results_file, 'r') as f:
        results = json.load(f)
    
    # Run evaluation
    evaluator = CorrectMSMARCOEvaluator()
    evaluator.evaluate_all_methods(results, args.output_dir)


if __name__ == '__main__':
    main()