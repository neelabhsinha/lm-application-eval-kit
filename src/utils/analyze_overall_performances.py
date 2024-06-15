import os
import textwrap

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from cycler import cycler

from const import results_dir, aggregated_results_dir, domains_filter, categories_filter, reasoning_filter
from src.utils.results_io_util import parse_name_to_dict
from src.utils.radar_chart_generator import RadarChartPlotter


def extract_aspects(string):
    result = string.split(', ')
    for i in range(len(result)):
        result[i] = result[i].split(' -> ')[0]
    return result


def sort_key(col):
    parts = col.split('_')
    example_num = int(parts[0])
    definition_status = 0 if 'with_definition' in col else 1
    explanation_status = 0 if 'with_explanation' in col else 1
    return definition_status, explanation_status, example_num


def get_stats(group_df):
    metrics = ['bleu', 'rouge1', 'rouge2', 'rougeL', 'meteor', 'bert_score_precision',
               'bert_score_recall', 'bert_score_f1']
    existing_metrics = [metric for metric in metrics if metric in group_df.columns]
    if existing_metrics:
        return group_df[existing_metrics].mean()
    else:
        return pd.Series([float('nan')] * len(metrics), index=metrics)


def filter_dataframe(df, values_list):
    filtered_df = df[df.index.isin(values_list)]
    filtered_df = filtered_df.loc[values_list]
    return filtered_df


def aggregate_aspect_level_results(file_path):
    columns_to_convert = ['categories', 'domains', 'reasoning', 'input_language', 'output_language',
                          'instruction_language']
    converters = {col: extract_aspects for col in columns_to_convert}
    df = pd.read_csv(file_path, converters=converters)
    domains_df = df.explode('domains').groupby('domains').apply(get_stats)
    categories_df = df.explode('categories').groupby('categories').apply(get_stats)
    reasoning_df = df.explode('reasoning').groupby('reasoning').apply(get_stats)
    return domains_df.reset_index(), categories_df.reset_index(), reasoning_df.reset_index()


class ModelPerformanceAnalysisUtil:
    def __init__(self, metric='bert_score', filter_categories=False, filter_domains=False, filter_reasoning=False,
                 save_markdown=True, write_all_aspect_level_results=False):
        self.result_folders = sorted(self._get_result_folders(), key=lambda x: x['model_name'])
        self.metric = metric
        self.save_markdown = save_markdown
        self.models = list({item['model_name'] for item in self.result_folders})
        self.write_all_aspect_level_results = write_all_aspect_level_results
        self.filter_categories = filter_categories
        self.filter_domains = filter_domains
        self.filter_reasoning = filter_reasoning

    @staticmethod
    def _get_result_folders():
        result_folders = os.listdir(results_dir)
        if '.DS_Store' in result_folders:
            result_folders.remove('.DS_Store')
        results_dict = list(map(parse_name_to_dict, result_folders))
        results_dict = [d for d in results_dict if
                        d.get('add_paraphrased_definition', False) == False and
                        d.get('add_adversarial_definition', False) == False and
                        d.get('do_sample', False) == False
                        ]
        return results_dict

    def _extract_model_level_performance_for_all_metrics(self):
        final_results = pd.DataFrame()
        for result_metadata in self.result_folders:
            df = pd.read_csv(os.path.join(results_dir, result_metadata['result_dir'], 'result_statistics.csv'),
                             index_col=0)
            metric_values = df.loc['mean']
            metric_values['best_instruction'] = str(result_metadata['positive_examples']) + ' examples ' + 'with' + (
                'out ' if not result_metadata['add_definition'] else ' ') + 'definition'
            if result_metadata['model_name'] not in final_results.columns:
                final_results[result_metadata['model_name']] = metric_values
            elif (final_results.loc['bert_score_recall', result_metadata['model_name']] <
                  metric_values['bert_score_recall']):
                final_results[result_metadata['model_name']] = metric_values
        final_results = final_results.T
        self._save_dataframe(final_results, os.path.join(aggregated_results_dir, 'results_on_all_metrics',
                                                         'comparison_across_all_metrics.csv'))

    def _extract_overall_model_level_performance(self):
        summary_dfs = [
            pd.read_csv(os.path.join(results_dir, result['result_dir'], 'result_statistics.csv'), index_col=0)
            for result in self.result_folders
        ]
        results = []
        for run_metadata, summary_df in zip(self.result_folders, summary_dfs):
            row_name = run_metadata['model_name']
            col_name = (str(run_metadata['positive_examples']) + '_examples_' +
                        ('with_definition' if run_metadata['add_definition'] else 'without_definition') +
                        ('_with_explanation' if run_metadata['add_explanation'] else '_without_explanation'))
            val = summary_df.loc['mean', self.metric].astype(float)
            results.append((row_name, col_name, val))
        results_df = pd.DataFrame(results, columns=['model_name', 'metadata', 'value'])
        pivot_df = results_df.pivot(index='model_name', columns='metadata', values='value')
        sorted_columns = sorted(pivot_df.columns, key=sort_key)
        pivot_df = pivot_df[sorted_columns]
        self._save_dataframe(pivot_df, os.path.join(aggregated_results_dir, self.metric, 'prompt_style_results_overall',
                                                    'overall_comparison_of_models_by_prompt_style.csv'))

    def _group_all_model_results_along_prompt(self):
        all_results = pd.DataFrame()
        for result_metadata in self.result_folders:
            if result_metadata['model_name'] not in self.models:
                continue
            predictions_dir = os.path.join(results_dir, result_metadata['result_dir'], 'predictions.csv')
            predictions = pd.read_csv(predictions_dir, index_col='prompt')
            predictions = predictions[[self.metric]]
            predictions.columns = [result_metadata['model_name']]
            if all_results.empty:
                all_results = predictions
            else:
                all_results = all_results.combine_first(predictions)
        self._find_correlation_across_different_models(all_results)
        if self.write_all_aspect_level_results:
            self._save_dataframe(all_results, os.path.join(aggregated_results_dir, self.metric,
                                                           'prompt_style_results_overall',
                                                           'result_by_prompt.csv'))

    def _find_correlation_across_different_models(self, all_results):
        path = os.path.join(aggregated_results_dir, self.metric, 'lm_performance_correlation')
        corr = all_results.corr()
        self._save_correlation_matrix(corr, os.path.join(path, 'generation_correlation_across_models_by_prompt.pdf'))

    def _extract_aspect_level_performance_by_model(self, model_name):
        model_dicts = [item for item in self.result_folders if item['model_name'] == model_name]
        domains_df_with_def = None
        domains_df_without_def = None
        categories_df_with_def = None
        categories_df_without_def = None
        reasoning_df_with_def = None
        reasoning_df_without_def = None
        for model_info in model_dicts:
            predictions_dir = os.path.join(results_dir, model_info['result_dir'], 'predictions.csv')
            add_definition = model_info['add_definition']
            positive_examples = model_info['positive_examples']
            domains_df, categories_df, reasoning_df = aggregate_aspect_level_results(predictions_dir)
            if self.write_all_aspect_level_results:
                self._save_individual_run_aspect_results(domains_df, categories_df, reasoning_df, model_info)
            if add_definition:
                domains_df_with_def = self._merge_dataframe(domains_df_with_def, domains_df, positive_examples,
                                                            'domains')
                categories_df_with_def = self._merge_dataframe(categories_df_with_def, categories_df, positive_examples,
                                                               'categories')
                reasoning_df_with_def = self._merge_dataframe(reasoning_df_with_def, reasoning_df, positive_examples,
                                                              'reasoning')
            else:
                domains_df_without_def = self._merge_dataframe(domains_df_without_def, domains_df, positive_examples,
                                                               'domains')
                categories_df_without_def = self._merge_dataframe(categories_df_without_def, categories_df,
                                                                  positive_examples, 'categories')
                reasoning_df_without_def = self._merge_dataframe(reasoning_df_without_def, reasoning_df,
                                                                 positive_examples, 'reasoning')
        domains_max, domains_min = self._get_max_min(domains_df_with_def, domains_df_without_def)
        categories_max, categories_min = self._get_max_min(categories_df_with_def, categories_df_without_def)
        reasoning_max, reasoning_min = self._get_max_min(reasoning_df_with_def, reasoning_df_without_def)
        self._write_aggregated_aspect_results_by_model(domains_df_with_def, 'domains', model_name,
                                                       domains_max, domains_min, 'domains_with_definition')
        self._write_aggregated_aspect_results_by_model(domains_df_without_def, 'domains', model_name,
                                                       domains_max, domains_min, 'domains_without_definition')
        self._write_aggregated_aspect_results_by_model(categories_df_with_def, 'categories', model_name,
                                                       categories_max, categories_min, 'categories_with_definition')
        self._write_aggregated_aspect_results_by_model(categories_df_without_def, 'categories', model_name,
                                                       categories_max, categories_min, 'categories_without_definition')
        self._write_aggregated_aspect_results_by_model(reasoning_df_with_def, 'reasoning', model_name,
                                                       reasoning_max, reasoning_min, 'reasoning_with_definition')
        self._write_aggregated_aspect_results_by_model(reasoning_df_without_def, 'reasoning', model_name,
                                                       reasoning_max, reasoning_min, 'reasoning_without_definition')
        best_results_df_domains = self._save_instruction_wise_performance_and_get_max_value(domains_df_with_def,
                                                                                            domains_df_without_def,
                                                                                            'domains', model_name)
        best_results_df_categories = self._save_instruction_wise_performance_and_get_max_value(categories_df_with_def,
                                                                                               categories_df_without_def,
                                                                                               'categories', model_name)
        best_results_df_reasoning = self._save_instruction_wise_performance_and_get_max_value(reasoning_df_with_def,
                                                                                              reasoning_df_without_def,
                                                                                              'reasoning', model_name)
        return best_results_df_domains, best_results_df_categories, best_results_df_reasoning

    def _save_instruction_wise_performance_and_get_max_value(self, df_with_def, df_without_def, aspect, model_name):
        if df_with_def is None or df_without_def is None:
            return None
        merged_df = pd.merge(df_with_def, df_without_def, on=aspect, how='outer',
                             suffixes=('_examples_with_definition', '_examples_without_definition'))
        numeric_cols = merged_df.select_dtypes(include=np.number).columns.tolist()
        sorted_columns = sorted(numeric_cols, key=sort_key)
        sorted_columns = [aspect] + sorted_columns
        merged_df = merged_df[sorted_columns]
        merged_df['max_value'] = merged_df[numeric_cols].max(axis=1)
        merged_df.loc[:, 'best_instruction'] = merged_df[numeric_cols].idxmax(axis=1)
        self._save_dataframe(merged_df, os.path.join(aggregated_results_dir, self.metric,
                                                     'aspect_prompt_style_variation',
                                                     model_name, aspect, 'all_results_with_best_instruction.csv'))
        merged_df = merged_df.rename(columns={'max_value': model_name})
        return merged_df[[aspect, model_name]]

    def _save_individual_run_aspect_results(self, domain_results, category_results, reasoning_results, run_metadata):
        row_name = run_metadata['model_name']
        col_name = (str(run_metadata['positive_examples']) + '_examples_' +
                    ('with_definition' if run_metadata['add_definition'] else 'without_definition') +
                    ('_with_explanation' if run_metadata['add_explanation'] else '_without_explanation'))
        folder_name = f'{row_name}/{col_name}'
        self._save_dataframe(domain_results,
                             os.path.join(aggregated_results_dir, self.metric, folder_name, 'domain_results.csv'))
        self._save_dataframe(category_results,
                             os.path.join(aggregated_results_dir, self.metric, folder_name, 'category_results.csv'))
        self._save_dataframe(reasoning_results,
                             os.path.join(aggregated_results_dir, self.metric, folder_name, 'reasoning_results.csv'))

    def _merge_dataframe(self, result_df, df, column_key, aspect):
        df = df[[aspect, self.metric]]
        df = df.rename(columns={self.metric: column_key})
        if result_df is None:
            result_df = df
        else:
            result_df = result_df.merge(df, on=aspect, how='outer')
        return result_df

    def _write_aggregated_aspect_results_for_pt_and_it_models(self, df, aspect):
        df = df.set_index(aspect)
        f = domains_filter if aspect == 'domains' else (categories_filter if aspect == 'categories'
                                                        else reasoning_filter)
        radar_chart_plotter = RadarChartPlotter()
        self._save_dataframe(df, os.path.join(aggregated_results_dir, self.metric, 'aspect_level_lm_analysis',
                                              f'comparison_across_{aspect}', 'results.csv'))
        radar_chart_plotter.plot_radar_chart(df, f, df.columns, os.path.join(aggregated_results_dir, self.metric,
                                                                             'aspect_level_lm_analysis',
                                                                             f'comparison_across_{aspect}',
                                                                             'radar_chart.pdf'))

    def _write_aggregated_aspect_results_by_model(self, df, aspect, model_name, max, min, file):
        order = domains_filter if aspect == 'domains' else (categories_filter if aspect == 'categories'
                                                            else reasoning_filter)
        if df is not None:
            df = df.set_index(aspect).loc[order].reset_index()
            columns_to_select = [aspect] + [0, 2, 4, 8]
            available_columns = [col for col in columns_to_select if col in df.columns]
            if available_columns:
                df = df[available_columns]
                folder_path = os.path.join(aggregated_results_dir, self.metric, 'aspect_prompt_style_variation',
                                           model_name, aspect, file)
                visualization_path = os.path.join(folder_path)
                all_columns_present = set(columns_to_select).issubset(df.columns)
                if all_columns_present:
                    self._save_line_graph(df, visualization_path, max, min, aspect)
            else:
                print(f"No available data columns from {columns_to_select} in DataFrame to process.")

    def _save_line_graph(self, df, file_path, max, min, aspect):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        plt.rcParams.update({'font.size': 22})
        color_cycle = plt.cm.tab20.colors
        fig, ax = plt.subplots(figsize=(16, 8))
        ax.set_facecolor('#e5ecf6')

        plt.rc('axes', prop_cycle=(cycler('color', color_cycle)))
        for index, row in df.iterrows():
            row = row.tolist()
            label = '\n'.join(textwrap.wrap(row[0], width=20))  # Adjust 'width' as needed
            plt.plot([0, 2, 4, 8], row[1:], marker='x', label=label, linewidth=2, markersize=10)

        plt.xlabel('Positive Examples')
        plt.ylabel(self.metric.replace('_', ' ').title())
        plt.ylim(min - 1, max + 1)
        plt.legend(loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0., fontsize=22)
        plt.grid(True, color='white', linestyle='-', linewidth=1.5)
        plt.tick_params(axis='both', which='major', labelsize=22)
        plt.tight_layout(rect=[0, 0, 0.75, 1])
        plt.savefig(file_path + '.pdf', bbox_inches='tight')
        plt.close()

    @staticmethod
    def _save_correlation_matrix(corr, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        plt.figure(figsize=(12, 10))
        ax = sns.heatmap(corr, annot=False, fmt=".2f", cmap='coolwarm',
                         xticklabels=corr.columns, yticklabels=corr.columns,
                         cbar_kws={"shrink": .75})
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=45, ha='right')
        plt.subplots_adjust(left=0.2, right=0.9, top=0.9, bottom=0.2)
        plt.savefig(path)
        plt.close()

    @staticmethod
    def _get_max_min(df_with_def, df_without_def):
        df_with_def_max = 100
        df_without_def_max = 100
        df_with_def_min = 0
        df_without_def_min = 0
        if df_with_def is not None:
            df_with_def_numeric = df_with_def.select_dtypes(include=[np.number])
            df_with_def_max = df_with_def_numeric.max().max()
            df_with_def_min = df_with_def_numeric.min().min()
        if df_without_def is not None:
            df_without_def_numeric = df_without_def.select_dtypes(include=[np.number])
            df_without_def_max = df_without_def_numeric.max().max()
            df_without_def_min = df_without_def_numeric.min().min()
        max_value = max(df_with_def_max, df_without_def_max)
        min_value = min(df_with_def_min, df_without_def_min)
        return max_value, min_value

    @staticmethod
    def _save_dataframe(df, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        df.to_csv(path)

    def get_results(self, models):
        if models is not None:
            self.models = models
        print('Collecting results for these models -', self.models)
        print('Extracting overall performances over all models')
        print('Extracting overall performances over all models')
        self._extract_model_level_performance_for_all_metrics()
        self._extract_overall_model_level_performance()
        print('Extracting performance of individual models across aspects')
        best_results_domains = None
        best_results_categories = None
        best_results_reasoning = None
        for model in self.models:
            best_results_df_domains, best_results_df_categories, best_results_df_reasoning = (
                self._extract_aspect_level_performance_by_model(model))
            if best_results_domains is None:
                best_results_domains = best_results_df_domains
            elif best_results_df_domains is not None:
                best_results_domains = best_results_domains.merge(best_results_df_domains, on='domains', how='outer')
            if best_results_categories is None:
                best_results_categories = best_results_df_categories
            elif best_results_df_categories is not None:
                best_results_categories = best_results_categories.merge(best_results_df_categories, on='categories',
                                                                        how='outer')
            if best_results_reasoning is None:
                best_results_reasoning = best_results_df_reasoning
            elif best_results_df_reasoning is not None:
                best_results_reasoning = best_results_reasoning.merge(best_results_df_reasoning, on='reasoning',
                                                                      how='outer')
        print('Extracting results for comparison of performance across aspects for all models')
        self._write_aggregated_aspect_results_for_pt_and_it_models(best_results_domains, 'domains')
        self._write_aggregated_aspect_results_for_pt_and_it_models(best_results_categories, 'categories')
        self._write_aggregated_aspect_results_for_pt_and_it_models(best_results_reasoning, 'reasoning')
        print('Analyzing Correlation along prompts for all models')
        self._group_all_model_results_along_prompt()
        print('Extracting pairwise performance aspects to determine best model and instruction')
