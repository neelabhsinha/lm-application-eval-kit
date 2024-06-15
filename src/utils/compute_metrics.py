from const import results_dir
from src.handler.exit_handler import EXIT
from src.utils.results_io_util import write_results
from src.metrics.bert_score import BertScore
from src.metrics.meteor import Meteor
from src.metrics.rouge import RougeScore
from src.metrics.bleu_score import BleuScore

import numpy as np
import os
import pandas as pd
from tqdm import tqdm


def compute_metrics(metric, force_recompute=False):
    skip_existing = False if force_recompute else True
    files = os.listdir(results_dir)
    if 'bleu' in metric:
        bleu_score_calculator = BleuScore()
    if 'bert_score' in metric:
        bert_score_calculator = BertScore()
    if metric == 'meteor':
        meteor_calculator = Meteor()
    if 'rouge' in metric:
        rouge_calculator = RougeScore()
    for file in tqdm(files, desc=f'Calculating {metric} for results'):
        path = os.path.join(results_dir, file, 'predictions.csv')
        try:
            df = pd.read_csv(path)
            predictions = df['candidate'].fillna('').tolist()
            references = df['reference'].fillna('').tolist()
            if metric == 'bleu' and (not skip_existing or 'bleu' not in df.columns):
                bleu = bleu_score_calculator.get_score(predictions, references)
                df['bleu'] = np.array(bleu) * 100
            if 'bert_score' in metric and (
                    not skip_existing or ('bert_score_recall' not in df.columns or 'bert_score_f1' not in df.columns
                                          or 'bert_score_precision' not in df.columns)):
                scores = bert_score_calculator.get_score(predictions, references)
                f1 = np.array(scores['f1']) * 100
                recall = np.array(scores['recall']) * 100
                precision = np.array(scores['precision']) * 100
                df['bert_score_precision'] = precision
                df['bert_score_recall'] = recall
                df['bert_score_f1'] = f1
            if metric == 'meteor' and (not skip_existing or 'meteor' not in df.columns):
                scores = meteor_calculator.get_score(predictions, references)
                scores = np.array(scores) * 100
                df['meteor'] = scores
            if 'rouge' in metric and (not skip_existing or (
                    'rouge1' not in df.columns or 'rouge2' not in df.columns or 'rougeL' not in df.columns)):
                scores = rouge_calculator.get_score(predictions, references)
                df['rouge1'] = np.array(scores['rouge1']) * 100
                df['rouge2'] = np.array(scores['rouge2']) * 100
                df['rougeL'] = np.array(scores['rougeL']) * 100
            write_results(df, os.path.join(results_dir, file))
        except FileNotFoundError:
            print(f'Prediction file {file} not found in the given folder')
        if EXIT.is_set():
            return
