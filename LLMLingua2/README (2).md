# LLMLingua2 MS MARCO Evaluation - CORRECTED

Clean system using **official MS MARCO evaluation scripts** (manually downloaded).

## ✅ CORRECTED Manual Setup

### 1. Create directory:
```bash
mkdir evaluation
```

### 2. Manual Download (4 files):
Go to: https://github.com/microsoft/MSMARCO-Question-Answering/tree/master/Evaluation

Download these files to `evaluation/` directory:
- **`ms_marco_eval.py`** ✅ **CORRECTED: Main evaluation script** 
- **`rouge.py`** - ROUGE calculations (required by main script)
- **`bleu.py`** - BLEU calculations (required by main script)  
- **`run.sh`** - Shell wrapper (optional)

### 3. Final structure:
```
evaluation/
├── ms_marco_eval.py      # ✅ CORRECTED: Main script
├── rouge.py              # ROUGE dependency
├── bleu.py               # BLEU dependency
└── run.sh                # Optional
```

### 4. Install packages:
```bash
pip install llmlingua datasets transformers torch requests
```

## Usage

```bash
# Test with 2 examples
python run_pipeline.py --num_examples 2

# Full evaluation (5000 examples)
python run_pipeline.py

# Separate phases (for GPU constraints)
python run_pipeline.py --phase compression --num_examples 100
python run_pipeline.py --phase evaluation --compression_results compression_results_TIMESTAMP.json
```

## What it does

1. **Compression**: LLMLingua2 with 3 methods + context analysis using separators
2. **API**: Gets responses from Gemini (gemini/gemini-2.0-flash)
3. **Evaluation**: Calls official MS MARCO `ms_marco_eval.py` script (CORRECTED)

## Output

```
compression_results_TIMESTAMP.json    # Compression + context analysis
msmarco_eval_TIMESTAMP/
├── msmarco_results.json              # MS MARCO evaluation outputs
├── references.json                   # Ground truth (MS MARCO format)
├── predictions_original.json         # Original predictions
├── predictions_method1_rate.json     # Rate method predictions
├── predictions_method2_target_tokens.json  # Token method predictions
└── predictions_method3_target_contexts.json # Context method predictions
```

## Files (corrected)

- `config.json` - Configuration (unchanged)
- `llmlingua_compressor.py` - Compression + context tracking (unchanged) 
- `msmarco_evaluator.py` - Corrected wrapper for `ms_marco_eval.py`
- `run_pipeline.py` - Corrected pipeline runner 

## ❌ Previous Error vs ✅ Correction

**❌ WRONG**: I said main file was `Q+A_Evaluation.py`  
**✅ CORRECT**: Main file is `ms_marco_eval.py`

**❌ WRONG**: Downloaded from wrong repository or wrong file  
**✅ CORRECT**: Download from https://github.com/microsoft/MSMARCO-Question-Answering/tree/master/Evaluation

## Dependencies

- `ms_marco_eval.py` **imports** functions from `rouge.py` and `bleu.py`
- All files must be in the **same directory** 
- Cannot download just one file - they work together as a set

## MS MARCO Format

**Input format for evaluation script**:
```json
{
  "query_id_1": "predicted answer text",
  "query_id_2": "another predicted answer"
}
```

**Reference format**:
```json
{
  "query_id_1": ["ground truth answer"],
  "query_id_2": ["another ground truth answer"]  
}
```

**Usage**: `python ms_marco_eval.py references.json predictions.json`

Thank you for the correction! Now using the proper `ms_marco_eval.py` script. 🎯